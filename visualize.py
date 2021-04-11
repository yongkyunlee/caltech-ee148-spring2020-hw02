import json
import os

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

DATA_PATH = './data/RedLights2011_Medium'
PREDS_PATH = './data/hw02_preds'
OUTPUT_PATH = './data/output'
os.makedirs(OUTPUT_PATH, exist_ok=True) # create directory if needed 

def visualize_boxes(file_names, preds, threshold):
    """ Visualize the bounding boxes to the source images """
    for file_name in tqdm(preds):
        img = Image.open(os.path.join(DATA_PATH, file_name))
        draw = ImageDraw.Draw(img)
        points_arr = preds[file_name]
        for points in points_arr:
            y0, x0, y1, x1, score = points
            if score > threshold:
                draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
        img.save(os.path.join(OUTPUT_PATH, file_name), "JPEG")

if __name__ == '__main__':
    preds_name = 'preds_train.json'

    file_names = sorted(os.listdir(DATA_PATH)) 
    file_names = [f for f in file_names if '.jpg' in f] 
    with open(os.path.join(PREDS_PATH, preds_name)) as fp:
        preds = json.load(fp)
    
    visualize_boxes(file_names, preds, 0.9)
