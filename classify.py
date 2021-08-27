import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, xywh2xyxy, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized

import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def clamp(x, a, b):
    return torch.minimum(torch.maximum(x, a), b)

def expand_mushroom(xyxy):
    x,y,w,h = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
    if h/w<2:
        hh = 2*w # new height
        y = y + (hh-h)//2 ## new y-center
        h = hh
    w = h
    return (xywh2xyxy(torch.tensor([x,y,w,h]).view(1, 4))).view(-1)

def expand_pole(xyxy, off=-0.05):
    x,y,w,h = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
    y += off*h
    h += 2*off*h
    return (xywh2xyxy(torch.tensor([x,y,w,h]).view(1, 4))).view(-1)

def scale_img(img, scale, pad=10):
    if scale<0:
        scale = -scale
        q = scale/min(img.shape[1:])
    else:
        q = scale/max(img.shape[1:])
    if q<1:
        h,w = int(img.shape[1]*q), int(img.shape[2]*q) 
        img = F.interpolate(img.unsqueeze(0), size=[h,w]).squeeze()
    if pad>0:
        img = F.pad(img, pad=(pad, pad, pad, pad), mode='constant', value=0)
    return img

def crop_one_box(x, img):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    return img[c1[1]:c2[1], c1[0]:c2[0]]

def load_classifier(model_file, fc, device):
    model = models.resnet34()
    num_ftrs = model.fc.in_features
    if fc==0:
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 2),
                                nn.LogSoftmax(dim=1))
    else:
        model.fc = nn.Sequential(nn.Linear(num_ftrs, fc),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Linear(fc, 2),
                                nn.LogSoftmax(dim=1))
    model.load_state_dict(torch.load(model_file))
    model.to(device).eval()
    return model

def fixname(str):
    return str.replace(' ','_').lower()

def load_classifiers(names, device):
    model_root = '/home/david/code/phawk/data/fpl/damage/rgb/resnet/python/models'
    models = adict()
    models[1] = adict({'name':fixname(names[1]), 'scale':-128, 'fc':512, 'func':expand_pole})   # Concrete Pole
    models[6] = adict({'name':fixname(names[6]), 'scale':256, 'fc':0, 'func':expand_mushroom})  # Mushroom Insulator
    models[7] = adict({'name':fixname(names[7]), 'scale':256, 'fc':64})    # Fuse Switch Polymer
    models[10] = adict({'name':fixname(names[10]), 'scale':256, 'fc':512})  # Porcelain Dead-end Insulator
    models[11] = adict({'name':fixname(names[11]), 'scale':256, 'fc':256})    # Porcelain Insulator
    models[16] = adict({'name':fixname(names[16]), 'scale':256, 'fc':128})  # Surge Arrester
    models[17] = adict({'name':fixname(names[17]), 'scale':256, 'fc':64})   # Transformer
    models[18] = adict({'name':fixname(names[18]), 'scale':512, 'fc':256})  # Wood Crossarm
    models[19] = adict({'name':fixname(names[19]), 'scale':-128, 'fc':256, 'func':expand_pole}) # Wood Pole
    models[20] = adict({'name':fixname(names[20]), 'scale':256, 'fc':256})  # Fuse Switch Porcelain

    for c in models.keys():
        models[c].model = load_classifier(f'{model_root}/{models[c].name}/{models[c].name}.pt', models[c].fc, device)
    return models

@torch.no_grad()
def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    # names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = True
    models = load_classifiers(names, device)  # initialize

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred,
                                    opt.conf_thres,
                                    opt.iou_thres,
                                    opt.classes,
                                    # opt.agnostic_nms, ## use default (True) instead
                                    max_det=opt.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        # pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    # Apply ResNet classifiers to cropped images
                    c = int(cls)
                    if c not in models:
                        continue
                    modelc = models[c].model
                    scale = models[c].scale
                    if 'func' in models[c]:
                        xyxy = models[c].func(xyxy)
                        xyxy = clamp(xyxy, gn*0, gn)
                    img_crop = crop_one_box(xyxy, imc)
                    ################
                    # label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                    # print(label)
                    # cv2.imwrite(f'/home/david/code/phawk/data/fpl/damage/rgb/detect/aaa.jpg', img_crop)
                    ################
                    img_crop = img_crop[:, :, ::-1] ## RGB->BGR (resnet models were trained on BGR inputs)
                    img_crop = torch.FloatTensor(img_crop.transpose(2,0,1)*1/255.)
                    img_crop = scale_img(img_crop, scale=scale)
                    ######################################
                    # mu = img_crop.view(3,-1).mean(1)
                    # print(f'{p.stem}')
                    # print(mu)
                    ######################################
                    prob_damage = modelc(torch.unsqueeze(img_crop, 0).to(device)).cpu().numpy()[1] #.argmax(1).cpu().numpy()

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)
