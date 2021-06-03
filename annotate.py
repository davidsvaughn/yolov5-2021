import argparse
import time
from pathlib import Path
import ntpath
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os,sys
import numpy as np
import glob
import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, xywhn2xyxy#(x, w=640, h=640,
from utils.plots import plot_one_box, color_list, crop_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def load_list(fn):
    with open(fn) as f:
        lines = f.read().splitlines()
    return lines

def load_labels(lab_file):
    with open(lab_file, 'r') as f:
        labels = [x.split() for x in f.read().strip().splitlines()]
    return np.array(labels, dtype=np.float32)

def path_parts(path):
    head, tail = ntpath.split(path)
    return head, tail.split('.')

def xywh2xyxy(xywh,W,H):
    cx,cy,w,h = xywh
    x1 = cx-w/2
    y1 = cy-h/2
    x2 = cx+w/2
    y2 = cy+h/2
    return np.array([x1*W,y1*H,x2*W,y2*H]).round().astype(np.int32)

def annotate(save_img=False):
    images, label_path, imgsz = opt.images, opt.labels, opt.img_size

    if not os.path.exists(images):
        print('{} not found!'.format(images))
        sys.exit()

    if not os.path.exists(label_path):
        print('{} not found!'.format(label_path))
        sys.exit()

    # get label filter (if any)
    filt = eval(opt.filter)
    filt = np.array(filt) if len(filt)>0 else None
    if filt is not None: prefilter = filt

    # get crop param (if any)
    crop = eval(opt.crop)
    crop = np.array(crop) if len(crop)>0 else None
    if crop is not None: prefilter = crop

    # filter images BEFORE loading them
    if prefilter is not None:
        lab_files = glob.glob(f'{label_path}/*.txt')
        img_files = glob.glob(f'{images}/*.[jJ][pP][gG]')
        images = []
        for lab_file in lab_files:
            labels = load_labels(lab_file)
            if len(labels)==0:
                continue
            y = np.array(labels)[:,0]
            idx = np.in1d(y, prefilter)
            if not idx.any():
                continue
            _, name = path_parts(lab_file)
            images.extend([f for f in img_files if '.'.join(name[:-1]) in f])
        random.shuffle(images)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()

    # Set Dataloader
    save_img = True
    dataset = LoadImages(images, img_size=imgsz)#, verbose=False)

    # Get names and colors
    names = load_list(opt.names)
    colors = color_list()

    ## process all images...
    n, ct = 0, 0
    for path, _, img, _ in dataset:
        ct += 1
        if ct%10==0: print(f'{ct}/{len(dataset)}')

        _, name = path_parts(path)
        lab_file = os.path.join(label_path, '{}.txt'.format('.'.join(name[:-1])))
        if not os.path.exists(lab_file):
            print('{} not found!'.format(lab_file))
            continue

        labels = load_labels(lab_file)
        if len(labels)==0:
            continue

        if filt is not None:
            y = np.array(labels)[:,0]
            idx = np.in1d(y,filt)
            if not idx.any():
                continue
            labels = labels[idx]

        if crop is not None:
            y = np.array(labels)[:,0]
            idx = np.in1d(y,crop)
            if not idx.any():
                continue
            labels = labels[idx]

        p = Path(path)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg

        ## resize image (unless cropping)...
        if crop is None:
            q = max(img.shape[:2])/imgsz
            h,w = int(img.shape[0]/q), int(img.shape[1]/q)
            img = cv2.resize(img, dsize=(w,h), interpolation=cv2.INTER_CUBIC)

        ## process all labels...
        first = names.copy() ## only print class name the first time
        for i,det in enumerate(labels):  # detections per image

            cls, xywh = det[0], det[1:]
            cls_id = int(cls)
            xyxy = xywh2xyxy(xywh, img.shape[1], img.shape[0])

            if crop is not None:
                n += 1
                crop_path = f'{save_dir}/{cls_id}-{i}_{n}_{p.name}'
                img_crop = crop_one_box(xyxy, img)
                cv2.imwrite(crop_path, img_crop)
                continue

            label = True
            if first and first[cls_id]:
                first[cls_id]=0
            else:
                label = False
            label = f'{names[cls_id]}' if label else None
            plot_one_box(xyxy, img, label=label, color=colors[cls_id % len(colors)], line_thickness=1)#3

        # Save results (image with detections)
        if crop is None:
            cv2.imwrite(save_path, img)

    # s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    print(f"Results saved to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='data/images', help='images')  # file/folder, 0 for webcam
    parser.add_argument('--labels', type=str, default='data/labels', help='labels')  # file/folder, 0 for webcam
    parser.add_argument('--names', default='coco.names', help='coco.names file')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--project', default='runs/annotate', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--filter', type=str, default='[]', help='label filter')
    parser.add_argument('--crop', type=str, default='[]', help='crop images')
    opt = parser.parse_args()
    print(opt)

    annotate()
