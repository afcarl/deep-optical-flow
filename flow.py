import cv2
import numpy as np

FARNEBACK_CONFIG = {
    'pyr_scale': 0.5,
    'levels': 3,
    'winsize': 15,
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma': 1.2,
    'flags': 0
}


def dense_flow(frame0, frame1, previous_flow=None):
    return cv2.calcOpticalFlowFarneback(frame0, frame1, flow=previous_flow,
                                 pyr_scale=FARNEBACK_CONFIG['pyr_scale'],
                                 levels=FARNEBACK_CONFIG['levels'],
                                 winsize=FARNEBACK_CONFIG['winsize'],
                                 iterations=FARNEBACK_CONFIG['iterations'],
                                 poly_n=FARNEBACK_CONFIG['poly_n'],
                                 poly_sigma=FARNEBACK_CONFIG['poly_sigma'],
                                 flags=FARNEBACK_CONFIG['flags'])


# from OpenCV Optical Flow sample
def draw_flow(img, flow, step=32, grayscale=False):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    if grayscale:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    cv2.polylines(vis, lines, 0, (0, 255, 0), thickness=2)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 2, (0, 255, 0), -1)
    return vis