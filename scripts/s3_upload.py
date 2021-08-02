import os, sys
import json
import random
from glob import glob
import boto3
import ntpath
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor

S3FILE = '/home/david/code/phawk/data/fpl/component/s3/s3urls.txt'
DST_DIR = '/home/david/code/phawk/data/fpl/component/s3/images'

IMGS = '/home/david/code/phawk/data/fpl/component/s3/images.txt'
LABS = '/home/david/code/phawk/data/fpl/component/s3/labels.txt'


#################################################################

s3_client = boto3.client('s3')
boto3.setup_default_session(profile_name='ph-ai-dev')  # To switch between different AWS accounts
jpg, txt = '.jpg', '.txt'

def load_list(fn):
    with open(fn) as f:
        lines = f.read().splitlines()
    return lines

def save_list(lst, fn):
    with open(fn, 'w') as f:
        for item in lst:
            f.write("%s\n" % item)
            
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext=''):
    pattern = os.path.join(path, f'*{ext}')
    return np.array([path_leaf(f) for f in glob(pattern)])
            
def download_s3_file(s3_url, dst_dir='.', filename=None, silent=False, overwrite=False):
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

def download_image(s3_url):
    return download_s3_file(s3_url, dst_dir=DST_DIR, silent=False)

def download_images(s3_urls):
    executor = ThreadPoolExecutor(max_workers=64)
    for s3_url in s3_urls:
        executor.submit(download_image, s3_url)
    executor.shutdown(wait=True)
    
    
def upload_s3_file(file_src, s3_path):
    filename = file_src.split("/").pop()
    ## fix filenames
    parts = filename.split('.')
    filename = parts[0]+'.'+parts[-1]
    ################
    try:
        bucket = s3_path.split("/")[0]
        key = s3_path.replace(f'{bucket}/', "")
        key = f'{key}/{filename}'
        with open(file_src, 'rb') as f:
            s3_client.upload_fileobj(f, bucket, key)
    except Exception as e:
        print(f"Failed to upload file to S3: {file_src}")
        # raise e

def upload_files(files, s3_path):
    executor = ThreadPoolExecutor(max_workers=32)
    for i,file_path in enumerate(files):
        executor.submit(upload_s3_file, file_path, s3_path)
        if i%100==0:
            print(f"{i}\t{file_path}")
    executor.shutdown(wait=True)        

########################################################

if __name__ == "__main__":
    
    img_path = '/home/david/code/phawk/data/fpl/thermal/images/'
    lab_path = '/home/david/code/phawk/data/fpl/thermal/labels/'
    
    img_files = get_filenames(img_path, jpg)
    lab_files = get_filenames(lab_path, txt)
    rgb = np.array([f.replace(jpg,'') for f in img_files])
    lab = np.array([f.replace(txt,'') for f in lab_files])
    
    idx = np.in1d(rgb, lab)
    extra_files = img_files[~idx]
    img_files = img_files[idx]
    
    xpath = '/home/david/code/phawk/data/fpl/thermal/images_extra/'
    for f in extra_files:
        src = img_path+f
        dst = xpath+f
        os.rename(src, dst)
    
    sys.exit()
    ##########
    
    s3_img_path = 'ai-labeling/FPL/thermal/aug2/images'
    img_files = [img_path+f for f in img_files]
    upload_files(img_files, s3_img_path)
    
    s3_lab_path = 'ai-labeling/FPL/thermal/aug2/labels'
    lab_files = [lab_path+f for f in lab_files]
    upload_files(lab_files, s3_lab_path)
    
    print('Done')
    sys.exit()
    
    #########################################
    
    # rgb_img_path = '/home/david/code/phawk/data/fpl/thermal/transfer/rgb/images/'
    rgb_img_path = '/home/david/code/phawk/data/fpl/thermal/transfer/rgb/annotate/exp/'
    rgb_lab_path = '/home/david/code/phawk/data/fpl/thermal/transfer/rgb/labels/'
    
    th_img_path = '/home/david/code/phawk/data/fpl/thermal/transfer/thermal/images/'
    th_lab_path = '/home/david/code/phawk/data/fpl/thermal/transfer/thermal/annotations/'
    
    ## upload RGB images
    # rgb_img_files = get_filenames(rgb_img_path, jpg)
    # # rgb_lab_files = get_filenames(rgb_lab_path, txt)
    # s3_path = 'ai-labeling/FPL/thermal/july6/rgb/images_labeled'
    # rgb_img_files = [rgb_img_path+f for f in rgb_img_files]
    # upload_files(rgb_img_files, s3_path)
    # print('Done')
    
    ## upload thermal images
    th_img_files = get_filenames(th_img_path, jpg)
    s3_path = 'ai-labeling/FPL/thermal/july6/thermal/images'
    th_img_files = [th_img_path+f for f in th_img_files]
    upload_files(th_img_files, s3_path)
    print('Done')
    
    sys.exit()
    
    #########################################
    ## upload labels
    # s3_path = 'ai-labeling/FPL/thermal/june25/rgb/labels'
    # rgb_lab_files = [rgb_lab_path+f for f in rgb_lab_files]
    # # file = rgb_lab_files[0]
    # # upload_s3_file(file, s3_path)
    # upload_files(rgb_lab_files, s3_path)
    # print('Done') 
    
    # th_img_files = get_filenames(th_img_path, jpg)
    # rgb = np.array([f.replace(jpg,'') for f in rgb_img_files])
    # lab = np.array([f.replace(txt,'') for f in rgb_lab_files])
    # idx = ~np.in1d(lab, rgb)
    # print(lab[idx])
    # for root in lab[idx]:
    #     labf = rgb_lab_path + root + txt
    #     if os.path.exists(labf):
    #         os.remove(labf)
    #         print(f'deleting: {labf}...')
    # sys.exit()
    
    # rgb = np.array([f.replace(jpg,'').split('.')[0] for f in rgb_img_files])
    # th = np.array([f.replace(jpg,'') for f in th_img_files])
    
    # idx1 = ~np.in1d(th, rgb)
    # th_roots = th[idx1]
    # for root in th_roots:
    #     imgf = th_img_path + root + jpg
    #     labf = th_lab_path + root + txt
    #     if os.path.exists(imgf):
    #         os.remove(imgf)
    #         print(f'deleting: {imgf}...')
    #     if os.path.exists(labf):
    #         os.remove(labf)
    #         print(f'deleting: {labf}...')
    
    # idx2 = ~np.in1d(rgb, th)
    # print(rgb[idx2])

###################################################3

    # lab_path = '/home/david/code/phawk/data/fpl/component/labels/'
    # new_lab_path = '/home/david/code/phawk/data/fpl/component/new_labels/'
    
    # labels = get_filenames(lab_path, txt)
    # new_labels = get_filenames(new_lab_path, txt)
    
    # idx = ~np.in1d(new_labels, labels)
    # newlabs = new_labels[idx]
    # newlabs.sort()
    # save_list(newlabs, '/home/david/code/phawk/data/fpl/component/new_labels.txt')
