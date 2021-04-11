import os
import math
import json
from multiprocessing import Pool

import numpy as np
from PIL import Image

def compute_convolution(I, T, stride=1):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)
    (T_rows, T_cols, _) = np.shape(T)
    # assume dimensions of T to be odd for simplicty
    assert T_rows % 2 == 1 and T_cols % 2 == 1

    # We want to keep the dimension of the heat map same as that of the image
    # If stride > 1, the pixel of the image that is not covered is left as 0
    # Also, since we do not pad (i.e. not consider red lights that are cut off)
    # we leave the edges as 0
    heatmap = np.zeros((n_rows, n_cols))
    # the center of template is calculated as (int(T_rows / 2), int(T_cols/2))
    for i in range(0, int(n_rows / 2), stride): # assume traffic light on the top half of the picture
        # skip if template goes off the image boundary
        if i - int(T_rows / 2) < 0 or i + int(T_rows / 2) >= n_rows:
            continue
        for j in range(0, n_cols, stride):
            if j - int(T_cols / 2) < 0 or j + int(T_cols / 2) >= n_cols:
                continue
            u_idx, d_idx = i - int(T_rows / 2), i + int(T_rows / 2)
            l_idx, r_idx = j - int(T_cols / 2), j + int(T_cols / 2)
            patch = I[u_idx:d_idx+1,l_idx:r_idx+1,:].reshape(-1)
            patch = patch / np.linalg.norm(patch)
            t = T.reshape(-1)
            t = t / np.linalg.norm(t)
            heatmap[i, j] = np.dot(patch, t)
    return heatmap


def predict_boxes(heatmap, t_row_map, t_col_map, threshold):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []
    n_rows, n_cols = heatmap.shape
    for i in range(int(n_rows / 2)):
        for j in range(n_cols):
            if heatmap[i][j] > threshold:
                t_rows, t_cols = t_row_map[i][j], t_col_map[i][j]
                tl_row, br_row = i - int(t_rows / 2), i + int(t_rows / 2)
                tl_col, br_col = j - int(t_cols / 2), j + int(t_cols / 2)
                output.append([tl_row, tl_col, br_row, br_col, heatmap[i][j]])

    return output


def detect_red_light_mf(I, T_arr, threshold):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
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
    # add resized templates
    for T in T_arr:
        # for scale in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        for scale in [0.45, 0.5, 0.55, 0.6]:
            T_scaled = scale_img(T, scale)
            template_arr.append(np.asarray(T_scaled))

    n_rows, n_cols, _ = I.shape
    t_row_map, t_col_map = np.zeros((n_rows, n_cols)), np.zeros((n_rows, n_cols))
    heatmap = np.zeros((n_rows, n_cols))
    for template in template_arr:
        t_rows, t_cols, _ = template.shape
        t_heatmap = compute_convolution(I, template)
        # update the maximum heapmap and corresponding template size
        for i in range(n_rows):
            for j in range(n_cols):
                if heatmap[i][j] < t_heatmap[i][j]:
                    heatmap[i][j] = t_heatmap[i][j]
                    t_row_map[i][j] = t_rows
                    t_col_map[i][j] = t_cols
    
    output = predict_boxes(heatmap, t_row_map, t_col_map, threshold)

    # remove overlapping bounding boxes
    remove_idx_arr = []
    for i in range(len(output) - 1):
        for j in range(i+1, len(output)):
            if i in remove_idx_arr or j in remove_idx_arr:
                continue
            y0_1, x0_1, y1_1, x1_1, score1 = output[i]
            y0_2, x0_2, y1_2, x1_2, score2 = output[j]
            dx = min(x1_1, x1_2) - max(x0_1, x0_2)
            dy = min(y1_1, y1_2) - max(y0_1, y0_2)
            if dx >= 0 and dy >= 0: # there is overlap
                if score1 < score2:
                    remove_idx_arr.append(i)
                else:
                    remove_idx_arr.append(j)
    remove_idx_arr = sorted(remove_idx_arr, reverse=True)
    for idx in remove_idx_arr:
        del output[idx]

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

def detect_worker(T_arr, threshold, file_name):
    """ Wrapper for multiprocessing """
    pass


if __name__ == '__main__':
    # Note that you are not allowed to use test data for training.
    # set the path to the downloaded data:
    data_path = './data/RedLights2011_Medium'

    # load splits: 
    split_path = './data/hw02_splits'
    file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
    file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

    # set a path for saving predictions:
    preds_path = './data/hw02_preds'
    os.makedirs(preds_path, exist_ok=True) # create directory if needed

    # Set this parameter to True when you're done with algorithm development:
    done_tweaking = False

    # templates
    T1 = Image.open('./red_light_single.jpg')
    T2 = Image.open('./red_light_double.jpg')

    print(T1.width, T1.height)
    print(T2.width, T2.height)

    '''
    Make predictions on the training set.
    '''
    preds_train = {}
    file_names_train = ['RL-011.jpg']
    for i in range(len(file_names_train)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_train[i]))
        print(I.width, I.height)

        # convert to numpy array:
        I = np.asarray(I)

        preds_train[file_names_train[i]] = detect_red_light_mf(I, [T1, T2], 0.8)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
        json.dump(preds_train,f)

    if done_tweaking:
        '''
        Make predictions on the test set. 
        '''
        preds_test = {}
        for i in range(len(file_names_test)):

            # read image using PIL:
            I = Image.open(os.path.join(data_path,file_names_test[i]))

            # convert to numpy array:
            I = np.asarray(I)

            preds_test[file_names_test[i]] = detect_red_light_mf(I)

        # save preds (overwrites any previous predictions!)
        with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
            json.dump(preds_test,f)
