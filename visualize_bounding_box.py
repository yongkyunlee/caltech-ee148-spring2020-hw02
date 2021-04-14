import json
import os

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

DATA_PATH = './data/RedLights2011_Medium'
PREDS_PATH = './data/hw02_preds'
OUTPUT_PATH = './data/output'
ANNOTATION_PATH = './data/hw02_annotations'
os.makedirs(OUTPUT_PATH, exist_ok=True) # create directory if needed 

def visualize_boxes(file_names, preds, threshold, annotations=None):
    """ Visualize the bounding boxes to the source images """
    for file_name in tqdm(preds):
        img = Image.open(os.path.join(DATA_PATH, file_name))
        draw = ImageDraw.Draw(img)
        points_arr = preds[file_name]
        for points in points_arr:
            y0, x0, y1, x1, score = points
            if threshold is None:
                draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
            elif score > threshold:
                draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
        if annotations is not None:
            gt_arr = annotations[file_name]
            for gt in gt_arr:
                draw.rectangle([gt[1], gt[0], gt[3], gt[2]], outline='blue', width=2)
        img.save(os.path.join(OUTPUT_PATH, file_name), "JPEG")

if __name__ == '__main__':
    preds_name = 'preds_train.json'

    file_names = sorted(os.listdir(DATA_PATH)) 
    file_names = [f for f in file_names if '.jpg' in f] 
    with open(os.path.join(PREDS_PATH, preds_name)) as fp:
        preds = json.load(fp)
    
    with open(os.path.join(ANNOTATION_PATH, 'annotations_train.json')) as fp:
        annotations = json.load(fp)

    visualize_boxes(file_names, preds, 0.92, annotations=annotations)
