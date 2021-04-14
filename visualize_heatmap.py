import os
import math

import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from run_predictions import compute_convolution

DATA_PATH = './data/RedLights2011_Medium'

def generate_red_light_heatmap(I, T_arr):
    def scale_img(img, scale):
        img_width, img_height = img.width * scale, img.height * scale
        if math.ceil(img_width) % 2 == 1:
            img_width = math.ceil(img_width)
        else:
            img_width = math.floor(img_width)
            if img_width % 2 == 0:
                img_width += 1
        if math.ceil(img_height) % 2 == 1:
            img_height = math.ceil(img_height)
        else:
            img_height = math.floor(img_height)
            if img_height % 2 == 0:
                img_height += 1
        return img.resize((img_width, img_height))

    template_arr = []
    for T in T_arr: # add resized templates
        for scale in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            T_scaled = scale_img(T, scale)
            template_arr.append(np.asarray(T_scaled))
    
    n_rows, n_cols, _ = I.shape
    t_row_map, t_col_map = np.zeros((n_rows, n_cols)), np.zeros((n_rows, n_cols))
    heatmap = np.zeros((n_rows, n_cols))
    for template in tqdm(template_arr):
        t_rows, t_cols, _ = template.shape
        t_heatmap = compute_convolution(I, template, stride=1, upper_portion=1)
        # update the maximum heapmap and corresponding template size
        for i in range(n_rows):
            for j in range(n_cols):
                if heatmap[i][j] < t_heatmap[i][j]:
                    heatmap[i][j] = t_heatmap[i][j]
                    t_row_map[i][j] = t_rows
                    t_col_map[i][j] = t_cols
    
    return heatmap

if __name__ == '__main__':
    # templates
    T1 = Image.open('./red_light_single.jpg')
    T2 = Image.open('./red_light_double.jpg')
    T1 = T1.resize((int(T1.width / 2), int(T1.height / 2)))
    T2 = T2.resize((int(T2.width / 2), int(T2.height / 2)))
    
    file_id = 'RL-011'
    I = Image.open(os.path.join(DATA_PATH, file_id + '.jpg'))
    I = np.asarray(I)

    if os.path.exists(file_id + '_heatmap.npy'):
        heatmap = np.load(file_id + '_heatmap.npy')
    else:
        heatmap = generate_red_light_heatmap(I, [T1, T2])
        np.save(file_id + '_heatmap.npy', heatmap)

    plt.figure()
    plt.imshow(np.array(Image.open(os.path.join(DATA_PATH, file_id + '.jpg'))))
    im = plt.imshow(heatmap, alpha=0.6)
    plt.colorbar(im)
    plt.savefig(file_id + '_heatmap.jpg')
