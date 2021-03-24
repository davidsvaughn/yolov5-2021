import os, sys
import json, random
import numpy as np
import glob
import ntpath
import pandas as pd
from collections import Counter
from skmultilearn.model_selection import IterativeStratification

## pip install scikit-multilearn
## pip install arff
## pip install tabulate

################################################################

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def read_lines(fn):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return lines

def get_labels(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    labs = np.array( [np.array(line.strip().split(' ')).astype(float).round(6) for line in lines])
    if len(labs)==0:
        return labs.tolist()
    labs[:,1:] = labs[:,1:].clip(0,1)
    labs = [[int(x[0])] + x[1:] for x in labs.tolist()]
    return labs

def label_matrix(lab_files):
    Y = []
    for lf in lab_files:
        f = LAB_DIR + lf
        labs = get_labels(f)
        Y.append([y[0] for y in labs])
    n = max([max(y) for y in Y if len(y)>0]) + 1
    Z = []
    for y in Y:
        z = np.zeros(n)
        for k,v in Counter(y).items():
            z[k] = v
        Z.append(z)
    return np.array(Z)

## perform binary (2-way) stratified split on multilabel data...
def binary_split(X, Y, split, order=2):
    split = np.array(split/split.sum())
    strat = IterativeStratification(order=order, n_splits=len(split), sample_distribution_per_fold=split.tolist())
    idx1, idx2 = next(strat.split(X, Y))
    ## switch if out of order with split...
    if np.sign(split[0]-split[1]) != np.sign(len(idx1)-len(idx2)):
        idx1, idx2 = idx2, idx1
    set1 = X[idx1, :], Y[idx1, :]
    set2 = X[idx2, :], Y[idx2, :]
    return set1, set2
    
def train_test_val_split(X, Y, splits, order=2):
    splits = np.array(splits)
    ## since strat_split only does binary (2-way) split
    ## do it twice to get train/test/val...
    split1 = np.array([splits[:2].sum(), splits[-1]])
    split2 = splits[:2]
    
    # X = X[:,None]
    ## first separate val_set (smallest set) from rest of the data
    data_set, val_set = binary_split(X, Y, split=split1, order=order)
    ## then separate data_set into train_set/test_set...
    train_set, test_set = binary_split(*data_set, split=split2, order=order)
    
    return train_set, test_set, val_set

def build_json_string(x, name='train'):
    label_file = LAB_DIR + x
    y = get_labels(label_file)
    s3url = S3_IMG_BUCKET + x.replace('.txt','.jpg')
        
    s = {'s3Url' : s3url,
         'annotations' : y,
         'datasets' : [name]
         }
    return json.dumps(s).replace("/", "\\/").replace(' ','')
    
def build_set(X_files, name='train'):
    entries = []
    for x in X_files.squeeze():
        e = build_json_string(x, name)
        entries.append(e)
    return entries

def build_manifest():
        
    ## load all labels
    lab_files = [path_leaf(f) for f in glob.glob('{}*.txt'.format(LAB_DIR))]
    random.shuffle(lab_files)
    X = np.array(lab_files)[:,None]
    Y = label_matrix(lab_files)
    
    ## get stratified splits
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = train_test_val_split(X, Y, SPLITS, order=2)
    
    ## convert to manifest entries
    train_entries = build_set(X_train, name='train')
    test_entries = build_set(X_test, name='test')
    val_entries = build_set(X_val, name='val')
    
    ## save manifest
    with open(MANIFEST_FILE, "w") as f:
        f.writelines('{}\n'.format(s) for s in train_entries)
        f.writelines('{}\n'.format(s) for s in test_entries)
        f.writelines('{}\n'.format(s) for s in val_entries)
    
    ## get class names
    nc = y_train.shape[1]
    if os.path.exists(NAMES_FILE):
        classes = np.array(read_lines(NAMES_FILE))
    else:
        classes = np.arange(nc)
    
    ## print stats
    df_image_counts = pd.DataFrame({'TRAIN': y_train.astype(np.bool).sum(0),
                                    'TEST': y_test.astype(np.bool).sum(0),
                                    'VAL': y_val.astype(np.bool).sum(0)}, 
                                   index = classes) 
    
    df_target_counts = pd.DataFrame({'TRAIN': y_train.astype(np.int32).sum(0),
                                     'TEST': y_test.astype(np.int32).sum(0),
                                     'VAL': y_val.astype(np.int32).sum(0)}, 
                                    index = classes)
    
    print('\nIMAGE COUNTS PER SET/CLASS:')
    print(df_image_counts.to_markdown())
    
    print('\nTARGET COUNTS PER SET/CLASS:')
    print(df_target_counts.to_markdown())
    
    print('\nIMAGE COUNTS PER SET:')
    print('TRAIN\t| {}'.format(y_train.shape[0]))
    print('TEST \t| {}'.format(y_test.shape[0]))
    print('VAL  \t| {}'.format(y_val.shape[0]))


if __name__ == "__main__":
    
    ## set these values appropriately...
    S3_IMG_BUCKET = 's3://ai-labeling/FPL/thermal/rgb_labeled/' ## where images will be stored
    DATA_DIR = '/home/david/code/repo/ai_docker/datasets/fpl/component/thermal_pairs/data/' 
    LAB_DIR = DATA_DIR + 'labels/' ## local path where all labels files are
    NAMES_FILE = DATA_DIR + 'coco.names' ## optional class names file
    MANIFEST_FILE = DATA_DIR + 'manifest.txt' ## where to save manifest locally
    
    ## TRAIN/TEST/VAL split... 
    ## ** don't need to normalize, just give *relative* weightings **
    ## the code will normalize so sum()==1
    SPLITS = [20,3,2]
    
    build_manifest()
    