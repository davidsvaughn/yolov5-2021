import os,sys
import numpy as np

def load_labels(lab_file):
    if not os.path.exists(lab_file):
        return None
    with open(lab_file, 'r') as f:
        labels = [x.split() for x in f.read().strip().splitlines()]
    return np.array(labels, dtype=np.float32).round(4)

## center+wh format to topleft+bottomright corner format
def xywh2xyxy(xywh):
    x,y,w,h = xywh[:,0],xywh[:,1],xywh[:,2],xywh[:,3]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    A = np.dstack([x1,y1,x2,y2]).squeeze().clip(0,1)
    if len(A.shape)==2: return A
    return A[None,:]

## accepts a single box and a list of boxes...
## returns array of iou values between 'box' and all elements in 'boxes'
def bbox_iou(box, boxes):  
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

label_file_1 = 'labels1.txt'
label_file_2 = 'labels2.txt'

labs1 = load_labels(label_file_1)
labs2 = load_labels(label_file_2)

classes1 = labs1[:,0].astype(np.int32)
classes2 = labs2[:,0].astype(np.int32)

boxes1 = xywh2xyxy(labs1[:,1:5])
boxes2 = xywh2xyxy(labs2[:,1:5])

print(boxes1)
print(boxes2)
print('')

## loop through each box in boxes1...
## and get array of iou values between box1 and all boxes in boxes2
for box1 in boxes1:
    ious = bbox_iou(box1.T, boxes2.T)
    print(ious)
print('')

## get matrix of iou values between all boxes in boxes1 and boxes2
M = np.array([bbox_iou(box1.T, boxes2.T) for box1 in boxes1])
print(M)