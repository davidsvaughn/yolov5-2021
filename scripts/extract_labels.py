import os, sys
import json
import random
import glob
import boto3
import ntpath
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
from json.decoder import JSONDecodeError
from manifest_entry import ManifestEntry
from inference_response import COCOCategory

LABEL_FILTER = None

## FPL RGB Damage
DATA_DIR        = '/home/david/code/repo/ai_docker/datasets/fpl/damage/rgb/aapril-13-yolo-L-damage/'
MANIFEST_FILE   = DATA_DIR + 'manifest.txt'
LAB_DIR         = DATA_DIR + 'labels'
IMG_DIR         = DATA_DIR + 'images'
# LABEL_FILTER = [0,2,4,8]

#-------------------------------------------------

if LABEL_FILTER is not None:
    LABEL_FILTER = np.array(LABEL_FILTER)

if not os.path.exists(LAB_DIR):
    os.makedirs(LAB_DIR)

def parse_manifest(manifest_path):
    # Initialize train/val/test sets.
    train_set = set()
    val_set = set()
    test_set = set()

    # Read in the manifest.
    manifest = []
    with open(manifest_path, 'r') as f:
        try:
            line_number = 0
            line = f.readline()
            while line is not None:
                line_json = json.loads(line)
                entry = ManifestEntry(json=line_json)
                if entry is not None:
                    entry.annotations = line_json['annotations']
                    manifest.append(entry)

                    # Split up the entries into train/val/test sets.
                    if "train" in entry.datasets:
                        train_set.add(entry)
                    if "val" in entry.datasets:
                        val_set.add(entry)
                    if "test" in entry.datasets:
                        test_set.add(entry)

                # Read the next line.
                line_number += 1
                line = f.readline()
        except JSONDecodeError:
            pass  # Do nothing. Reached end of file.
    print(f'Finished reading manifest: {len(manifest)} lines.')
    return list(train_set), list(val_set), list(test_set), manifest

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def filter_labels(img_set):
    new_set = []
    for entry in img_set:
        if len(entry.annotations)==0:
            continue
        y = np.array(entry.annotations)[:,0]
        if not np.in1d(y, LABEL_FILTER).any():
            continue
        new_set.append(entry)
    return new_set
        
def write_labels(img_set, lab_dir=LAB_DIR, overwrite=True):
    for entry in img_set:
        label_file_path = f'{lab_dir}/{entry.resource_name()}.txt'
        if os.path.exists(label_file_path):
            if overwrite:
                os.remove(label_file_path)
            else:
                continue
        with open(label_file_path, 'a') as f:
            for annotation in entry.annotations:
                class_id, x, y, w, h = annotation
                f.write(f'{class_id} {x} {y} {w} {h}\n') 

###################################################################

if __name__ == "__main__":
    
    _, _, _, img_set = parse_manifest(MANIFEST_FILE)
    
    if LABEL_FILTER is not None:
        img_set = filter_labels(img_set)
        
    ## save image filenames to file
    write_labels(img_set)
