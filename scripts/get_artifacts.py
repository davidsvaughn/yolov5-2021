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
S3_FILTER = None
'''
Use this file to download saved model artifacts (and data) after a training run... 
It will arrange the files in the same structure as 'training_start.py' script in ai_docker.
'''
''' Set the following 3 file paths appropriately.... This example is for FPL Thermal Damage... '''


## FPL RGB Component - july8
DATA_DIR        = '/home/david/code/phawk/data/fpl/component'
MODEL_DIR       = '/home/david/code/phawk/data/fpl/component/models/july8'
MODEL_BUCKET    = 's3://ai-inference-dev-model-catalog/model/yolo-v5-full-scale/fpl-comp-1408-3008-yolov5l6-july8/'


#-----------------------------------------------
IMG_DIR = '{}/images'.format(DATA_DIR)
LAB_DIR = '{}/labels'.format(DATA_DIR)
if MODEL_BUCKET:
    MANIFEST_URL    = MODEL_BUCKET + 'manifest.txt'
    MODEL_WTS_URL   = MODEL_BUCKET + 'weights.pt'
    CFG_URL         = MODEL_BUCKET + 'hyp.yaml'
    CAT_URL         = MODEL_BUCKET + 'categories.json'
    TESTLOG_URL     = MODEL_BUCKET + 'test.txt'

#-------------------------------------------------

s3_client = boto3.client('s3')
boto3.setup_default_session(profile_name='ph-ai-dev')  # To switch between different AWS accounts

if LABEL_FILTER is not None:
    LABEL_FILTER = np.array(LABEL_FILTER)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

if not os.path.exists(LAB_DIR):
    os.makedirs(LAB_DIR)

def download_s3_file(s3_url, dst_dir=DATA_DIR, filename=None, silent=False, overwrite=False):
    if filename is None:
        filename = s3_url.split("/").pop()
    out_path = os.path.join(dst_dir, filename)
    if os.path.exists(out_path) and not overwrite:
        return out_path 
    try:
        if not silent:
            print(f'Downloading file from: {s3_url} to {out_path}')
        s3_url_components = s3_url.replace("s3://", "")
        bucket = s3_url_components.replace("s3://", "").split("/")[0]
        key = s3_url_components.replace(f'{bucket}/', "")
        with open(out_path, 'wb') as f:
            s3_client.download_fileobj(bucket, key, f)
        return out_path
    except Exception as e:
        # Print the URL of the failed file and re-raise the exception.
        print(f"Failed to download file from S3: {s3_url}")
        # raise e

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

def parse_categories(categories_path):
    categories = []
    with open(categories_path, 'r') as f:
        categories_dict = json.loads(f.read())
    cats = categories_dict['categories']
    cats = sorted(cats, key=lambda i: i['id'])
    for cat in cats:
        categories.append(COCOCategory(json=cat))
    print(f'Finished reading categories: {len(categories)} classes.')
    return categories

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def download_image(entry: ManifestEntry):
    s3_url = entry.s3Url
    return download_s3_file(s3_url, dst_dir=IMG_DIR, silent=True)

def download_images(img_set):
    executor = ThreadPoolExecutor(max_workers=64)
    for entry in img_set:
        executor.submit(download_image, entry)
    executor.shutdown(wait=True)

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

def filter_s3files(img_set):
    new_set = []
    for entry in img_set:
        for s in S3_FILTER:
            if s in entry.s3Url:
                new_set.append(entry)
                break
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

def write_set(set_path, img_set):
    if os.path.exists(set_path):
        os.remove(set_path)
    with open(set_path, 'a') as f:
        for entry in img_set:
            f.write(f'{IMG_DIR}/{entry.image_file_name()}\n')
            
def write_coco_names(categories, dst_dir=DATA_DIR):
    coco_names_filename = f'{dst_dir}/coco.names'
    if os.path.exists(coco_names_filename):
        os.remove(coco_names_filename)
    labels = []
    with open(coco_names_filename, 'a') as f:
        for category in categories:
            name = category.name
            labels.append(name)
            f.write(f'{name}\n')
    return labels

def write_data_yaml(labels, dst_dir=DATA_DIR):
    data_yaml_filename = f"{dst_dir}/data.yaml"
    if os.path.exists(data_yaml_filename):
        os.remove(data_yaml_filename)
    with open(data_yaml_filename, 'a') as f:
        # f.write(f"train: {DATA_DIR}/train.txt\n")
        f.write(f"val: {dst_dir}/val.txt\n")
        f.write(f"test: {dst_dir}/test.txt\n\n")
        f.write(f'# Number of classes in dataset:\nnc: {len(labels)}\n\n')
        f.write(f'# Class names:\nnames: {json.dumps(labels)}')

###################################################################

if __name__ == "__main__":
    
    manifest_path = download_s3_file(MANIFEST_URL, dst_dir=MODEL_DIR, overwrite=False)# True
    train_set, val_set, test_set, all_set = parse_manifest(manifest_path)
    
    ## remove leakage!!!
    train = [ntpath.split(e.s3Url)[1] for e in train_set]
    val = [ntpath.split(e.s3Url)[1] for e in val_set]
    test = [ntpath.split(e.s3Url)[1] for e in test_set]
    idx_val = np.where(~np.in1d(val, train))[0]
    idx_test = np.where(~np.in1d(test, train))[0]
    val_set = [val_set[i] for i in idx_val]
    test_set = [test_set[i] for i in idx_test]
    # sys.exit()
    
    if CAT_URL:
        categories_path = download_s3_file(CAT_URL, dst_dir=MODEL_DIR, overwrite=True)
        categories = parse_categories(categories_path)
        labels = write_coco_names(categories, dst_dir=MODEL_DIR)
        write_data_yaml(labels, dst_dir=MODEL_DIR)
        
    if MODEL_BUCKET:
        model_path = download_s3_file(MODEL_WTS_URL, dst_dir=MODEL_DIR, overwrite=False)# True
        cfg_path = download_s3_file(CFG_URL, dst_dir=MODEL_DIR, overwrite=True)
        download_s3_file(TESTLOG_URL, dst_dir=MODEL_DIR, filename='testlog.txt', overwrite=True)
    
    img_sets = []
    img_sets.append((test_set, 'test'))
    img_sets.append((val_set, 'val'))
    img_sets.append((train_set, 'train')) ## skipping train set.... usually very large
    
    for img_set in img_sets:
        img_set, name = img_set
        # random.shuffle(img_set)
        
        if LABEL_FILTER is not None:
            img_set = filter_labels(img_set)
        
        if S3_FILTER is not None:
            img_set = filter_s3files(img_set)
        
        print(f'Downloading {name} set... {len(img_set)} images...')
        download_images(img_set)
        print('...done.')
        
        ## filter test manifest entries on downloaded images
        img_files = [path_leaf(f) for f in glob.glob('{}/*.[jJ][pP][gG]'.format(IMG_DIR))]
        img_set = [e for e in img_set if e.image_file_name() in img_files]
        
        ## save image filenames to file
        write_labels(img_set)
        img_set_path = f'{MODEL_DIR}/{name}.txt'
        write_set(img_set_path, img_set)
