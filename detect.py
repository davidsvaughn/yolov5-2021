import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.class_corrections import get_corrected_class, hide_container_object
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, box_ios
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


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
    if half:
        model.half()  # to FP16

    # Hotspot Model (optional)
    thermal = opt.thermal or opt.hotspot
    if opt.hotspot:
        hmodel = attempt_load(opt.hotspot, map_location=device)  # load FP32 model
        if half:
            hmodel.half()  # to FP16
        names.append('Hotspot') ## add extra label
    
    # (Contrast Limited) Adaptive Histogram Equalization (....done at model training time)
    if thermal:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

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
        if opt.hotspot:
            hmodel(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(hmodel.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:

        if thermal:
            ## apply Adaptive Histogram Equalization (CLAHE)
            himg = img.copy() if opt.hotspot else img
            # s = f'img{np.random.randint(0,1000)}'
            img = img.transpose(1,2,0)
            # cv2.imwrite(f'/home/product/dvaughn/data/fpl/thermal/stuff/{s}.jpg', img)
            img = clahe.apply(img[:,:,0].astype(np.uint8))[:,:,None] * np.ones(3, dtype=int)[None, None, :]
            # cv2.imwrite(f'/home/product/dvaughn/data/fpl/thermal/stuff/{s}_clahe.jpg', img)
            img = img.transpose(2,0,1)
            im0s = (clahe.apply(im0s[:,:,0].astype(np.uint8))[:,:,None] * np.ones(3, dtype=int)[None, None, :]).astype(np.uint8)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        # Component Inference
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, max_det=opt.max_det)

        # Hotspot Inference
        if opt.hotspot:
            himg = torch.from_numpy(himg).to(device)
            himg = himg.half() if half else himg.float()  # uint8 to fp16/32
            himg /= 255.0  # 0 - 255 to 0.0 - 1.0
            if himg.ndimension() == 3:
                himg = himg.unsqueeze(0)
            hpred = hmodel(himg, augment=opt.augment)[0]
            hpred = non_max_suppression(hpred, opt.hconf_thres, opt.iou_thres)
        t2 = time_synchronized()

        # Process detections
        first = names.copy() if names else None
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            ##### testing123......
            # if '2b731fab-e1b3-3ab1-b0fe-0db2e896f512' in Path(p).stem:
            #     print('stop')

            # Find hotspot-component overlaps
            if opt.hotspot:
                dh1 = None
                hdet = hpred[i]
                if len(det) and len(hdet):
                    boxes = det[:, :4]
                    spots = hdet[:, :4]
                    ## ios == "Intersection-Over-Smaller" (for when box1 is *inside* box2.... gives high score)
                    ios = box_ios(spots, boxes).cpu().numpy()
                    ios = (ios > opt.ios_thres)
                    dh1 = (ios.sum(0)>0).astype(np.int32)  # components that do overlap with a hotspot
                    hd0 = (ios.sum(1)==0) # hotspots that don't overlap with a component
                elif len(hdet):
                    hd0 = np.ones(len(hdet))>0
                elif len(det):
                    dh1 = np.zeros(len(det))

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                n = len(det)
                for i, (*xyxy, conf, cls) in enumerate(reversed(det)):

                    # Correct some classes if they're inside a bounding box of a particular class?
                    if opt.correct_classes:
                        cls = get_corrected_class(cls=cls, xyxy=xyxy, det=det, names=names)

                    # If it's not a container object for corrected classes, that needs to be hidden,
                    # write detection to file, image, etc.
                    if not (opt.correct_classes and opt.hide_containers and hide_container_object(cls)):

                        hot = dh1[n-i-1] if (opt.hotspot and dh1 is not None) else None

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            if opt.hotspot:
                                line = (cls, *xywh, conf, hot) if opt.save_conf else (cls, *xywh, hot)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or opt.save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = True
                            if first and first[c]:
                                first[c]=0
                            else:
                                label = False
                            name = f'{names[c]} ' if label else ''
                            label = None if opt.hide_labels else (name if opt.hide_conf else f'{name}{conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                            if opt.save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            ## hotspot labels 
            if opt.hotspot and len(hdet):
                # Rescale boxes from img_size to im0 size
                hdet[:, :4] = scale_coords(img.shape[2:], hdet[:, :4], im0.shape).round()

                # Write results
                n = len(hdet)
                N = len(names)-1
                for i, (*xyxy, conf, cls) in enumerate(reversed(hdet)):
                    no_overlap = hd0[n-i-1] ## does hotspot overlap with component?

                    # only write label if no overlap
                    if no_overlap and save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls+N, *xywh, conf, 0) if opt.save_conf else (cls+N, *xywh, 0)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    # draw boxes for all hotspots
                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls+N)  # integer class
                        label = True
                        if first and first[c]:
                            first[c]=0
                        else:
                            label = False
                        name = f'{names[c]} ' if label else ''
                        label = None if opt.hide_labels else (name if opt.hide_conf else f'{name}{conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness, yoff=1, xoff=1)
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
    parser.add_argument('--weights', type=str, help='component model weights')
    parser.add_argument('--thermal', action='store_true', help='thermal images')
    parser.add_argument('--hotspot', default='', type=str, help='hotspot model weights')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='component confidence threshold')
    parser.add_argument('--hconf-thres', type=float, default=0.1, help='hotspot confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='IOU threshold for NMS')
    parser.add_argument('--ios-thres', type=float, default=0.2, help='IOS threshold for hotspot-component overlap')
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
    parser.add_argument('--line-thickness', default=0, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--correct-classes', action='store_true', help='correct classes found inside certain objects')
    parser.add_argument('--hide-containers', action='store_true', help='hide objects containing corrected classes')
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)
