import os
import cv2
import flow
import numpy as np
import persistence

ACTIONS = {'LEFT': 'l', 'RIGHT': 'r', 'STOP': 's', 'GO': 'g'}
DEFAULT_INPUT = "sample2.mp4"
THRESHOLD = 0.3
DEBUG = False
ROI_PERCENT = 0.1


def draw_label(frame, position, label):
    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)


def gray_scale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def roi_vertices(shape):
    x_horizon, y_horizon = (shape[0] * ROI_PERCENT, shape[1] * ROI_PERCENT)
    x_half, y_half = (shape[0] / 2, shape[1] / 2)

    top_left = (y_half + y_horizon, 0)
    top_right = (y_half - y_horizon, 0)

    bottom_right = (y_half - y_horizon, shape[0])
    bottom_left = (y_half + y_horizon, shape[0])

    return np.array([[top_right, top_left, bottom_left, bottom_right]], dtype=np.int32)


def define_roi(frame):
    mask = np.zeros_like(frame)

    if len(frame.shape) > 2:
        channel_count = frame.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, roi_vertices(frame.shape), ignore_mask_color)
    masked_frame = cv2.bitwise_and(frame, mask)

    return masked_frame


def label_frame(frame0, frame1, pre_label0=ACTIONS['GO']):
    label1 = ACTIONS['GO']

    roi_frame0 = define_roi(frame0)
    roi_frame1 = define_roi(frame1)

    dense_flow = flow.dense_flow(frame0=gray_scale(roi_frame0), frame1=gray_scale(roi_frame1))
    dx, dy = np.split(dense_flow, 2, axis=2)
    mean_dx = np.mean(dx)

    # super Naive.. :-D
    if mean_dx > THRESHOLD:
        label1 = ACTIONS['LEFT']
    elif mean_dx < -THRESHOLD:
        label1 = ACTIONS['RIGHT']

    return label1, mean_dx, dense_flow


def process(video_source, output_folder):
    capture = cv2.VideoCapture(video_source)

    flag, frame0 = capture.read()
    if flag:
        index0 = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
        label0 = ACTIONS['GO']
        persistence.save_label(frame0, label0, index0, output_folder)

    while flag:
        flag, frame1 = capture.read()
        index1 = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
        if flag:
            label1, mean_dx, dense_flow = label_frame(frame0, frame1, pre_label0=label0)
            persistence.save_label(frame1, label1, index1, output_folder, dense_flow=dense_flow, annotate=True)
            frame0 = frame1
            label0 = label1


process(DEFAULT_INPUT, output_folder=os.path.splitext(DEFAULT_INPUT)[0])


