import argparse
import time
from pathlib import Path
import glob, os, sys
import cv2
import json
import yaml
import ntpath
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models
import torch.nn.functional as F
import ast

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, xywhn2xyxy, xywh2xyxy, strip_optimizer, set_logging, increment_path, save_one_box, box_iou, box_io1
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
img_formats += [f.upper() for f in img_formats] # add upper case versions

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def aslist(x):
    try:
        _ = (e for e in x)
    except TypeError:
        return [x]
    return x

def is_grayscale(img):
    return np.mean(img[...,0] - img[...,1]) < 0.001

def init_clahe(cliplimit=3.0, dim=8):
    return cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(dim,dim))

def apply_clahe(clahe, img):#if self.clahe is not None and is_grayscale(img):
    return clahe.apply(img[:,:,0].astype(np.uint8))[:,:,None] * np.ones(3, dtype=int)[None, None, :]

def read_lines(fn):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return lines

def empty_file(fn):
    with open(fn, 'w') as f:
        f.write('')

def load_inference_image(image_path, scale, stride):
    img0 = cv2.imread(image_path) if isinstance(image_path, str) else image_path
    if img0 is None:
        return None, None, None
    # Pad & resize
    img, ratio, pad = letterbox(img0, new_shape=scale, stride=stride, scaleup=False)
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img, img0, (ratio, pad)

def numpy2cuda(img, device, half=False):
    img = torch.from_numpy(img).to(device).float()
    if half:
        img = img.half()
    img /= 255.0  # 0-255 --> 0.0-1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0) ## make it a batch of size 1
    return img

class FileIterator:
    def __init__(self, path):
        if isinstance(path, list):
            files = path
        else:
            p = str(Path(path).absolute())  # os-agnostic absolute path
            if '*' in p:
                files = sorted(glob.glob(p, recursive=True))  # glob
            elif os.path.isdir(p):
                files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
            elif os.path.isfile(p) and p.endswith('.jpg'):
                files = [p]  # files
            elif os.path.isfile(p) and p.endswith('.txt'):
                files = read_lines(p)
            else:
                raise Exception(f'ERROR: {p} does not exist')
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        ni = len(images)
        self.files = images 
        self.nf = ni

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        self.count += 1
        return path

class YoloModel:
    def __init__(self, weights_path, scale, conf_thres, iou_thres, device, cct, sem_classes=None, sem_conf_thres=0.1, sem_iou_thres=0.1, opt=None):
        self.weights_path = weights_path
        self.scale = scale
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.sem_classes = sem_classes
        self.sem_conf_thres = sem_conf_thres
        self.sem_iou_thres = sem_iou_thres
        self.device = device
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if cct is not None: ## class confidence thresholds
            if isinstance(cct, str):
                cct = ast.literal_eval(cct)
            cct = None if len(cct)==0 else torch.from_numpy(np.array(cct)).to(device)
        self.cct = cct
        # self.opt = opt
        self.init_model()

    def init_model(self):
        self.model = attempt_load(self.weights_path, map_location=self.device) # load FP32 model
        if self.half:
            self.model.half()  # to FP16
        self.stride = int(self.model.stride.max())  # model stride
        self.scale = check_img_size(self.scale, s=self.stride)  # check scale
        # run once
        self.model(torch.zeros(1, 3, self.scale, self.scale).to(self.device).type_as(next(self.model.parameters())))  # run once
        # Load the class names.
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

    @torch.no_grad()
    def run(self, img, classes=None):
        preds = self.model(img)[0] # augment=opt.augment

        if self.sem_classes is None: ## default --> normal one-shot NMS
            pred = non_max_suppression(preds, self.conf_thres, self.iou_thres, classes, cct=self.cct)
        else: ## split NMS between normal object classes and semantic classes (roads, fences, rivers, etc...)
            nc = preds.shape[2] - 5  ## get total number of classes
            obj_classes = list(set(np.arange(nc)) - set(self.sem_classes)) ## list normal object classes
            ## get normal object predictions
            pred_obj = non_max_suppression(preds, self.conf_thres, self.iou_thres, classes=obj_classes, cct=self.cct)
            ## get semantic class predictions
            pred_sem = non_max_suppression(preds, self.sem_conf_thres, self.sem_iou_thres, classes=self.sem_classes, cct=self.cct)
            ## recombine predictions
            pred = torch.cat((pred_obj[0], pred_sem[0]), 0)
            pred = [pred]

        return pred

class Detector:
    def __init__(self, 
                weights=None, 
                img_size=None, 
                conf_thres=None, 
                iou_thres=None, 
                device=None, 
                classes=None, 
                categories_path=None, 
                cct=None, 
                sem_classes=None, 
                sem_conf_thres=0.1, 
                sem_iou_thres=0.1,
                clahe=None,
                yolargs=None,
                hyp=None,
                opt=None):

        if opt is not None:
            weights, img_size, conf_thres, iou_thres, device, classes, categories_path, cct, sem_classes, sem_conf_thres, sem_iou_thres, hyp = \
                opt.weights, opt.img_size, opt.conf_thres, opt.iou_thres, opt.device, opt.classes, opt.categories, opt.cct, opt.sem_classes, opt.sem_conf_thres, opt.sem_iou_thres, opt.hyp
        
        ## parse yolargs parameters....
        try:
            if yolargs is not None and len(yolargs)>0:
                hyp = yolargs

            if hyp is not None and len(hyp)>0:
                ## hyp should evaluate to a python dict()
                hyp = ast.literal_eval(hyp)
                if 'scale' in hyp:
                    img_size = hyp['scale']
                if 'threshold' in hyp:
                    conf_thres = hyp['threshold']
                if 'conf_thres' in hyp:
                    conf_thres = hyp['conf_thres']
                if 'iou_thres' in hyp:
                    iou_thres = hyp['iou_thres']
                if 'sem_conf_thres' in hyp:
                    sem_conf_thres = hyp['sem_conf_thres']
                if 'sem_iou_thres' in hyp:
                    sem_iou_thres = hyp['sem_iou_thres']
                if 'sem_classes' in hyp:
                    sem_classes = hyp['sem_classes']
                if 'cct' in hyp:
                    cct = hyp['cct']
                if 'clahe' in hyp:
                    clahe = hyp['clahe']
        except:
            print(f'ERROR: problem parsing yolargs string: {hyp}')

        self.classes = classes
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'
        self.model, self.stride, self.scale = self.init_model(weights, img_size, conf_thres, iou_thres, cct, sem_classes, sem_conf_thres, sem_iou_thres, opt)
        self.clahe = init_clahe(*aslist(clahe)) if clahe else None

        # Load the categories/class-names
        # categories_path = categories
        if categories_path is not None:
            with open(categories_path) as categories_file:
                categories_json = json.load(categories_file)['categories']
                self.names = []
                for category_json in categories_json:
                    self.names.append(category_json['name'])
        else:
            self.names = self.model.names

    def init_model(self, weights, img_size, conf_thres, iou_thres, cct, sem_classes, sem_conf_thres, sem_iou_thres, opt=None):
        model = YoloModel(weights, img_size, conf_thres, iou_thres, self.device, cct, sem_classes, sem_conf_thres, sem_iou_thres, opt)
        return model, model.stride, model.scale
    
    def detect(self, img_file):
        ## prepare image
        img, img0, ratio_pad = load_inference_image(img_file, scale=self.scale, stride=self.stride)
        if img0 is None:
            return None, None

        #### CLAHE #############################################
        if self.clahe is not None and is_grayscale(img0):
            ## apply to model input image
            img = img.transpose(1,2,0)
            img = apply_clahe(self.clahe, img)
            img = img.transpose(2,0,1)

            ## apply clahe to output image also???
            img0 = apply_clahe(self.clahe, img0).astype(np.uint8)
        #########################################################

        ## run model
        pred = self.model.run(numpy2cuda(img, self.device, self.half), self.classes)
        ## process detections
        det = pred[0] ## batch size is always 1 for detect (for now....)
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        detections = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[1:], det[:, :4], img0.shape, ratio_pad).round()
            # loop thru detections
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                detections.append((xywh, xyxy, conf, cls))
        return detections, img0

    def get_detections(self, img_file):
        return self.detect(img_file)

    def format_labels(self, xywh, cls, conf, opt):
        # if thermal hotspot model, conf *may* be a tuple (conf, v), where v is extra value...
        if isinstance(conf, tuple):
            conf = conf[0]
        txt_lab = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
        img_lab = '' if opt.hide_conf else f'{conf:.2f}'
        return txt_lab, img_lab

    def run_detections(self, opt):
        image_files = FileIterator(opt.source)
        
        for k,img_file in enumerate(image_files):
            p = Path(img_file)  # to Path

            if opt.save_dir is None:
                print('save_dir is set to None!')
                sys.exit()
            t1 = time_synchronized()
            detections, img0 = self.get_detections(img_file)#, opt)
            t2 = time_synchronized()

            if k%10==0: print(k)
            if opt.skip_empty and len(detections)==0:
                continue

            # Print results to screen(?)
            s = '%gx%g ' % img0.shape[:2]  # print string
            cls_idx = np.array([det[-1].cpu().numpy() for det in detections])
            for c in np.unique(cls_idx):
                n = (cls_idx == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
            save_path = str(opt.save_dir / p.name)
            txt_path = str(opt.save_dir / 'labels' / p.stem)
            first = self.names.copy() if self.names else None

            # Loop over detections
            imc = img0.copy() if opt.save_crop else img0  # for opt.save_crop
            for xywh, xyxy, conf, cls in detections:
                
                # Write label to txt file
                txt_lab, img_lab = self.format_labels(xywh, cls, conf, opt)
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(txt_lab)).rstrip() % txt_lab + '\n')
                
                # Add bbox to image
                if opt.save_img or opt.save_crop:
                    c = int(cls)  # integer class
                    label = True
                    if first and first[c]:
                        first[c]=0
                    else:
                        label = False
                    name = f'{self.names[c]} ' if label else ''
                    label = None if opt.hide_labels else f'{name}{img_lab}'
                    plot_one_box(xyxy, img0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                    if opt.save_crop:
                        save_one_box(xyxy, imc, file=opt.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)
            
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            if opt.save_img and len(detections)>0:
                cv2.imwrite(save_path, img0)

            if len(detections)==0:
                empty_file(txt_path + '.txt')
                
    def detect_and_annotate(self, img_in):
        detections, img_out = self.get_detections(img_in)
        # Loop over detections
        first = self.names.copy() if self.names else None
        for xywh, xyxy, conf, cls in detections:
            if isinstance(conf, tuple):
                conf = conf[0]
            img_lab = f'{conf:.2f}'
            # Add bbox to image
            c = int(cls)  # integer class
            label = True
            if first and first[c]:
                first[c]=0
            else:
                label = False
            name = f'{self.names[c]} ' if label else ''
            label = f'{name}{img_lab}'
            plot_one_box(xyxy, img_out, label=label, color=colors(c, True), line_thickness=6)
        return img_out


class ThermalDetector(Detector):
    def __init__(self, 
                weights=None, 
                img_size=None, 
                conf_thres=None, 
                iou_thres=None, 
                hweights=None, 
                hconf_thres=0.5, 
                hiou_thres=0.2, 
                device=None, 
                classes=None, 
                categories_path=None, 
                cct=None, 
                opt=None,
                yolargs=None, 
                PA_mode=False):

        super(ThermalDetector, self).__init__(weights, img_size, conf_thres, iou_thres, device, classes, categories_path, cct, opt, yolargs)
        if opt is not None:
            hweights, img_size, hconf_thres, hiou_thres, cct = opt.hweights, opt.img_size, opt.hconf_thres, opt.hiou_thres, opt.cct

        ## parse yolargs parameters....
        try:
            if yolargs is not None and len(yolargs)>0:
                ## yolargs should evaluate to a python dict()
                yolargs = ast.literal_eval(yolargs)
                if 'scale' in yolargs:
                    img_size = yolargs['scale']
                if 'hconf_thres' in yolargs:
                    hconf_thres = yolargs['hconf_thres']
                if 'hiou_thres' in yolargs:
                    hiou_thres = yolargs['hiou_thres']
        except:
            print(f'ERROR: problem parsing yolargs string: {yolargs}')

        self.PA_mode = PA_mode
        ## initialize hotspot model...
        self.hmodel, _, _ = self.init_model(hweights, img_size, hconf_thres, hiou_thres, cct, opt)
        ## if coming from inference stack, supply categories file with "Hotspot" already at the end
        if not PA_mode:
            self.names.append('Hotspot') ## add extra label for Hotspot class
        ## initialize Adaptive Histogram Equalization (CLAHE)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    
    ## overriding base class...
    def get_detections(self, img_file, opt=None):
        if opt is not None:
            return self.detect(img_file, opt.hide_normal, opt.hide_solo)
        return self.detect(img_file)
    
    ## overriding base class...
    def format_labels(self, xywh, cls, conf, opt):
        # if thermal or damage model, conf is a tuple (conf, v), where v is extra value...
        if not opt.hide_normal:
            conf, hot = conf
            txt_lab = (cls, *xywh, conf, hot) if opt.save_conf else (cls, *xywh, hot)
        else:
            txt_lab = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
        img_lab = '' if opt.hide_conf else f'{conf:.2f}'
        return txt_lab, img_lab

    ## overriding base class...
    def detect(self, img_file, hide_normal=True, hide_solo=True):
        ## prepare image
        img, img0, ratio_pad = load_inference_image(img_file, scale=self.scale, stride=self.stride)
        if img0 is None:
            return None, None
        
        ## apply Adaptive Histogram Equalization (CLAHE)
        himg = img.copy() ## hotspot model does NOT use CLAHE
        img = img.transpose(1,2,0)
        img = self.clahe.apply(img[:,:,0].astype(np.uint8))[:,:,None] * np.ones(3, dtype=int)[None, None, :]
        img = img.transpose(2,0,1)
        img0 = (self.clahe.apply(img0[:,:,0].astype(np.uint8))[:,:,None] * np.ones(3, dtype=int)[None, None, :]).astype(np.uint8)

        ## run both models (thermal component and thermal hotspot)
        pred = self.model.run(numpy2cuda(img, self.device, self.half), self.classes)
        hpred = self.hmodel.run(numpy2cuda(himg, self.device, self.half))

        ## find component-hotspot overlaps
        det, hdet = pred[0], hpred[0]
        dh1 = None
        if len(det) and len(hdet):
            boxes = det[:, :4]
            spots = hdet[:, :4]
            ious = box_io1(spots, boxes).cpu().numpy()
            mask = (ious >= self.hmodel.iou_thres).astype(np.int32)
            ious = ious * mask

            # find best match for each hotspot
            col_idx = np.argmax(ious, 1)
            row_idx = np.arange(len(col_idx))
            mask = mask*0
            mask[row_idx, col_idx] = np.ones(len(col_idx))
            ious = ious * mask

            dh1 = (ious.sum(0)>0).astype(np.int32)  # components that do overlap with a hotspot
            hd0 = (ious.sum(1)==0) # hotspots that don't overlap with a component
        elif len(hdet):
            hd0 = np.ones(len(hdet))>0
        elif len(det):
            dh1 = np.zeros(len(det))
    
        # select only components that match with a hotspot
        if hide_normal:
            try:
                idx = dh1>0
                det, dh1 = det[idx], dh1[idx]
            except:
                pass

        ## process components
        detections = []
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[1:], det[:, :4], img0.shape, ratio_pad).round()
            # loop thru detections
            n = len(det)
            for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                if not hide_normal:
                    hot = dh1[n-i-1] if (dh1 is not None) else 0
                    conf = (conf, hot) # hot is a binary indicator for component having a hotspot
                detections.append((xywh, xyxy, conf, cls))
        
        ## process all hotspots and (maybe) add to detections
        if len(hdet) and not self.PA_mode:
            hdet[:, :4] = scale_coords(img.shape[1:], hdet[:, :4], img0.shape, ratio_pad).round()
            n = len(hdet)
            cls = torch.Tensor([len(self.names)-1]) ## cls for Hotspot
            for i, (*xyxy, conf, _) in enumerate(reversed(hdet)):
                no_overlap = hd0[n-i-1] ## hotspot does NOT overlap with any component
                if no_overlap and hide_solo:
                    continue
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                if not hide_normal:
                    conf = (conf, 1 if no_overlap else 0) # binary hotspot indicator always 1 for solo hotspot
                detections.append((xywh, xyxy, conf, cls))

        return detections, img0

@torch.no_grad()
def detect(opt):
    set_logging()
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    opt.save_dir = save_dir
    opt.save_img = not opt.nosave

    if opt.hotspot:
        detector = ThermalDetector(opt=opt)
    else:
        detector = Detector(opt=opt)

    t0 = time.time()
    detector.run_detections(opt)
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='component model weights')
    parser.add_argument('--categories', type=str, help='component model classes')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='component confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', type=str, default=None, help='filter by class: --class [0], or --class [0, 2, 3]')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=0, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--skip-empty', default=False, action='store_true', help='dont print txt or img if no labels')
    parser.add_argument('--cct', type=str, default='[]', help='class confidence thresholds')
    parser.add_argument('--hotspot', default='', type=str, help='hotspot model weights')
    parser.add_argument('--hconf-thres', type=float, default=0.5, help='hotspot confidence threshold')
    parser.add_argument('--hiou-thres', type=float, default=0.2, help='IOU threshold for hotspot-component overlap')
    parser.add_argument('--hide-normal', default=False, action='store_true', help='hide normal components, if hotspot')
    parser.add_argument('--hide-solo', default=False, action='store_true', help='hide solo/unmatched hotspots')
    parser.add_argument('--sem-classes', type=str, default=None, help='list semantic classes: --sem-class [0], or --sem-class [0, 2, 3]')
    parser.add_argument('--sem-iou-thres', type=float, default=0.1, help='sematic class IOU threshold for NMS')
    parser.add_argument('--sem-conf-thres', type=float, default=0.1, help='semantic class confidence threshold')
    parser.add_argument('--hyp', type=str, default=None, help='dict of hyperparameters')

    opt = parser.parse_args()
    if opt.classes is not None:
        opt.classes = eval(opt.classes)
    if opt.sem_classes is not None:
        opt.sem_classes = eval(opt.sem_classes)
    if opt.hyp is not None:
        opt.hyp = opt.hyp.replace(';',':').replace('\\','')
    print(opt)
    # check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)
