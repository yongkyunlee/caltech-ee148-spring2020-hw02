import os
import json
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    y0_1, x0_1, y1_1, x1_1 = box_1
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    y0_2, x0_2, y1_2, x1_2 = box_2
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)

    dx = min(x1_1, x1_2) - max(x0_1, x0_2)
    dy = min(y1_1, y1_2) - max(y0_1, y0_2)
    
    if dx > 0 and dy > 0:
        iou = dx * dy / (area1 + area2 - dx * dy)
    else:
        iou = 0
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    for file_name in preds:
        pred = preds[file_name]
        gt = gts[file_name]
        pred_cnt = len(list(filter(lambda x: x[4] >= conf_thr, pred)))
        pred_positive = set([])
        for i in range(len(gt)):
            n_detection = 0
            for j in range(len(pred)):
                if j in pred_positive: # a prediction cannot be compared to two ground truths
                    continue
                if pred[j][4] >= conf_thr: # only consider predictions with high enough confidence
                    iou = compute_iou(pred[j][:4], gt[i])
                    if iou >= iou_thr: # correct prediction
                        pred_positive.add(j)
        TP += len(pred_positive)
        FN += max(len(gt) - len(pred_positive), 0)
        FP += pred_cnt - len(pred_positive)

    return TP, FP, FN

def plot_PR_curve(tp_arr, fp_arr, fn_arr, title, file_name):
    precision = tp_arr / (tp_arr + fp_arr)
    recall = tp_arr / (tp_arr + fn_arr)
    plt.figure()
    plt.scatter(recall[0,:], precision[0,:])
    plt.scatter(recall[1,:], precision[1,:])
    plt.scatter(recall[2,:], precision[2,:])
    plt.title(title)
    plt.legend(['IoU Threshold 0.25', 'IoU Threshold 0.5', 'IoU Threshold 0.75'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(file_name)

if __name__ == '__main__':
    # set a path for predictions and annotations:
    preds_path = './data/hw02_preds'
    gts_path = './data/hw02_annotations'

    # load splits:
    split_path = './data/hw02_splits'
    file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
    file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

    # Set this parameter to True when you're done with algorithm development:
    done_tweaking = True

    # load train data
    with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
        preds_train = json.load(f)
    with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
        gts_train = json.load(f)

    if done_tweaking:
        # load test data      
        with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
            preds_test = json.load(f)
        with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
            gts_test = json.load(f)


    # For a fixed IoU threshold, vary the confidence thresholds.
    # The code below gives an example on the training set for one IoU threshold. 
    confidence_thrs = []
    for file_name in preds_train:
        confidence_thrs += list(map(lambda x: float(x[4]), preds_train[file_name]))
    confidence_thrs = np.sort(np.array(confidence_thrs)) # using (ascending) list of confidence scores as thresholds
    # store value for IOU threshold 0.25, 0.5, and 0.75
    tp_train = np.zeros((3, len(confidence_thrs))) 
    fp_train = np.zeros((3, len(confidence_thrs)))
    fn_train = np.zeros((3, len(confidence_thrs)))
    for i in tqdm(range(len(confidence_thrs))):
        conf_thr = confidence_thrs[i]
        tp_train[0, i], fp_train[0, i], fn_train[0, i] = compute_counts(preds_train, gts_train, iou_thr=0.25, conf_thr=conf_thr)
        tp_train[1, i], fp_train[1, i], fn_train[1, i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)
        tp_train[2, i], fp_train[2, i], fn_train[2, i] = compute_counts(preds_train, gts_train, iou_thr=0.75, conf_thr=conf_thr)
    
    plot_PR_curve(tp_train, fp_train, fn_train, 'PR curve for the train set',
                  'PR_curve_train.png')

    if done_tweaking:
        confidence_thrs = []
        for file_name in preds_test:
            confidence_thrs += list(map(lambda x: float(x[4]), preds_test[file_name]))
        confidence_thrs = np.sort(np.array(confidence_thrs)) # using (ascending) list of confidence scores as thresholds
        # store value for IOU threshold 0.25, 0.5, and 0.75
        tp_test = np.zeros((3, len(confidence_thrs))) 
        fp_test = np.zeros((3, len(confidence_thrs)))
        fn_test = np.zeros((3, len(confidence_thrs)))
        for i in tqdm(range(len(confidence_thrs))):
            conf_thr = confidence_thrs[i]
            tp_test[0, i], fp_test[0, i], fn_test[0, i] = compute_counts(preds_test, gts_test, iou_thr=0.25, conf_thr=conf_thr)
            tp_test[1, i], fp_test[1, i], fn_test[1, i] = compute_counts(preds_test, gts_test, iou_thr=0.5, conf_thr=conf_thr)
            tp_test[2, i], fp_test[2, i], fn_test[2, i] = compute_counts(preds_test, gts_test, iou_thr=0.75, conf_thr=conf_thr)
        
        plot_PR_curve(tp_test, fp_test, fn_test, 'PR curve for the test set',
                      'PR_curve_test.png')
