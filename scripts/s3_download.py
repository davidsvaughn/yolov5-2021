import os, sys
import json
import random
import glob
import boto3
import ntpath
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor

S3FILE = '/home/david/code/phawk/data/fpl/component/s3/s3urls.txt'
DST_DIR = '/home/david/code/phawk/data/fpl/component/images'

IMGS = '/home/david/code/phawk/data/fpl/component/images.txt'
LABS = '/home/david/code/phawk/data/fpl/component/s3/labels.txt'


#################################################################

s3_client = boto3.client('s3')
boto3.setup_default_session(profile_name='ph-ai-dev')  # To switch between different AWS accounts

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
    
def load_list(fn):
    with open(fn) as f:
        lines = f.read().splitlines()
    return lines

def save_list(lst, fn):
    with open(fn, 'w') as f:
        for item in lst:
            f.write("%s\n" % item)

######################################

# imgs = load_list(IMGS)
# labs = load_list(LABS)
# sys.exit()

# imgs = np.array([img.split('.')[0] for img in imgs])
# labs = np.array([lab.split('.')[0] for lab in labs])
# imgs.sort()
# labs.sort()

# ## find missing files
# idx = ~np.in1d(labs, imgs)
# miss = labs[idx]

# prefix = np.unique(['_'.join(s.split('_')[:2]) for s in miss])
# print(prefix)

# s3_urls = []
# s3bucket = 's3://ai-labeling/FPL/hardening-ca/cycle_4/jpg'
# for s in miss:
#     i = s.index('_')
#     s = f'{s3bucket}/{s[:i]} {s[i+1:]}.jpg'
#     s3_urls.append(s)

# save_list(s3_urls, S3FILE)

######################################

# imgs = load_list(IMGS)
# s3bucket = 's3://ai-labeling/FPL/components/May18/images'
# s3_urls = [f'{s3bucket}/{f}' for f in imgs]

# # s3_urls = load_list(S3FILE)
# download_images(s3_urls)
# print('Done.')
