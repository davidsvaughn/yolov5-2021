import os,sys
import ntpath, glob
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import numpy as np
from utils.general import xywh2xyxy
from utils.metrics import ap_per_class, ConfusionMatrix
import torch
import itertools 
import random
from random import shuffle as shuf

jpg, txt = '.jpg', '.txt'

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=jpg):
    pattern = os.path.join(path, f'*{ext}')
    return np.array([path_leaf(f) for f in glob.glob(pattern)])

def read_lines(fn):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return lines

def load_labels(lab_file):
    if not os.path.exists(lab_file):
        return None
    with open(lab_file, 'r') as f:
        labels = [x.split() for x in f.read().strip().splitlines()]
    return np.array(labels, dtype=np.float32).round(6)

def box_iou(box, boxes):  
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(box[0], boxes[0])
    yA = np.maximum(box[1], boxes[1])
    xB = np.minimum(box[2], boxes[2])
    yB = np.minimum(box[3], boxes[3])
    
    interW = xB - xA
    interH = yB - yA
    
    # Correction: reject non-overlapping boxes
    z = (interW>0) * (interH>0)
    interArea = z * interW * interH
    
    boxAArea = (box[2] - box[0]) * (box[3] - box[1])
    boxBArea = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def bbox_iou(boxes1, boxes2): 
    return np.array([box_iou(box1.T, boxes2.T) for box1 in boxes1])

#############################################################

iou_thres = 0.25
max_by_class = False

## Rich 70
# pred_dir = '/home/product/dvaughn/data/fpl/component/rich/yolov5_testing/rich_metrics/pred'
# lab_dir = '/home/product/dvaughn/data/fpl/component/rich/yolov5_testing/rich_metrics/truth'

## RGB 604
pred_dir = '/home/product/dvaughn/data/fpl/component/models/latest/detect/run_3008/labels'
lab_dir = '/home/product/dvaughn/data/fpl/component/labels'

pred_files = get_filenames(pred_dir, txt)
# shuf(pred_files)

class_file = '/home/product/dvaughn/data/fpl/component/rich/yolov5_testing/classes.txt'
names = read_lines(class_file)
nc = len(names)
iouv = np.arange(iou_thres, 1, 0.05)
niou = len(iouv)

ni = np.zeros(nc)
stats = []
for fn in pred_files:
    # fn = 'RaptorGuard_PRIMAVISTA_405533_RGB_54.txt'
    # print(fn)
    #####################
    lf = f'{lab_dir}/{fn}'
    pf = f'{pred_dir}/{fn}'
    labels = load_labels(lf)
    pred = load_labels(pf)
    nl = len(labels)
    tcls = labels[:, 0].tolist() if nl else [] 
    correct = np.zeros([pred.shape[0], niou], dtype=bool)
    
    if nl:
        detected = []  # target indices
        tcls_tensor = labels[:,0].astype(np.int32)
        ni[np.unique(tcls_tensor)] += 1
        tbox = xywh2xyxy(labels[:, 1:5])
        pbox = xywh2xyxy(pred[:, 1:5])
        
        # if plots:
        #     confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

        # Per target class
        for cls in np.unique(tcls_tensor):
            ti = (cls == tcls_tensor).nonzero()[0] # target indices
            pi = (cls == pred[:, 0]).nonzero()[0] # prediction indices
            # Search for detections
            if pi.shape[0]:
                boxes1 = pbox[pi]
                boxes2 = tbox[ti]
                ious = bbox_iou(boxes1, boxes2)
                i = ious.argmax(1)
                ious = ious[np.arange(ious.shape[0]), i]
                # Append detections
                detected_set = set()
                for j in (ious > iouv[0]).nonzero()[0]:
                    d = ti[i[j]]  # detected target
                    if d not in detected_set:
                        detected_set.add(d)
                        detected.append(d)
                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                        if len(detected) == nl:  # all targets already located in image
                            break

    stats.append((correct, pred[:,5], pred[:,0], tcls))
stats = [np.concatenate(x, 0) for x in zip(*stats)]

mp, mr, map50, map, mf1, ap_class, conf_best, nt, (p, r, ap50, ap, f1, cc) = ap_per_class(*stats,
                                                                                            names=names,
                                                                                            max_by_class=max_by_class, 
                                                                                            # conf_thres=conf_thres,
                                                                                            # plot=plots, 
                                                                                            # save_dir=save_dir, 
                                                                                            # ct=ct, 
                                                                                            )

# Print results
fmt = '%{}s'.format(2+max([len(s) for s in names]))
s = (fmt + '%12s' * 7) % ('Class', 'Images', 'Targets', 'P', 'R', 'F1', 'mAP@.5', 'mAP@.5:.95')
pf = fmt + '%12.3g' * 7  # print format
print(f'\n{s}')
print(pf % ('all', len(pred_files), nt.sum(), mp, mr, mf1, map50, map))
for i, c in enumerate(ap_class):
    print(pf % (names[c], ni[c], nt[c], p[i], r[i], f1[i], ap50[i], ap[i]))
if conf_best>-1:
    print('\nOptimal Confidence Threshold: {0:0.3f}'.format(conf_best))
    if max_by_class:
        print('Optimal Confidence Thresholds (Per-Class): {}'.format(list(cc.round(3))))

print('\nDone.')