import os
import fnmatch
import cv2
import numpy as np
import persistence

ANNOTATION_RETRIEVE_FORMAT = "{:s}/{:05d}.a.jpeg"
FRAME_RETRIEVE_FORMAT = "{:s}/{:05d}.?.o.jpeg"
DEFAULT_INPUT = "sample0.mp4"
SCALE = 0.8


def draw_label(frame, position, label):
    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)


def display_review(frame0_info, frame1_info):
    frame0_o, frame0_a = cv2.resize(frame0_info[0], None, fx=SCALE, fy=SCALE), cv2.resize(frame0_info[2], None, fx=SCALE, fy=SCALE)
    frame1_o, frame1_a = cv2.resize(frame1_info[0], None, fx=SCALE, fy=SCALE), cv2.resize(frame1_info[2], None, fx=SCALE, fy=SCALE)

    frame0_name = frame0_info[1]

    analyzing_stack = np.hstack((frame0_o, frame0_a))
    next_stack = np.hstack((frame1_o, frame1_a))

    draw_label(analyzing_stack, (20,20), frame0_name)

    cv2.imshow("review", np.vstack((analyzing_stack, next_stack)))



def retrieve_frame_info(index, input_folder):
    annotation_filename_pattern = ANNOTATION_RETRIEVE_FORMAT.format(input_folder, index)
    frame_filename_pattern = FRAME_RETRIEVE_FORMAT.format(input_folder, index)

    annotation, annotation_filename = None, None
    frame, frame_filename = None, None

    for dir_entry in os.scandir(input_folder):
        if fnmatch.fnmatch(dir_entry.path, annotation_filename_pattern):
            annotation = cv2.imread(dir_entry.path)
            annotation_filename = dir_entry.path
        if fnmatch.fnmatch(dir_entry.path, frame_filename_pattern):
            frame = cv2.imread(dir_entry.path)
            frame_filename = dir_entry.path

    return frame, frame_filename, annotation, annotation_filename


def manual_review(video_source, input_folder):
    capture = cv2.VideoCapture(video_source)
    frame_size = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for index in range(1, frame_size):
        frame0_info = retrieve_frame_info(index, input_folder)
        frame1_info = retrieve_frame_info(index + 1, input_folder)
        display_review(frame0_info, frame1_info)

        read_key = 0xFF & cv2.waitKey()
        if read_key == 27:
            break
        elif read_key == ord('l'):
            persistence.save_label(frame=frame0_info[0], label='l', index=index, output_folder=input_folder)
            os.remove(frame0_info[1])
        elif read_key == ord('r'):
            persistence.save_label(frame=frame0_info[0], label='r', index=index, output_folder=input_folder)
            os.remove(frame0_info[1])
        elif read_key == ord('g'):
            persistence.save_label(frame=frame0_info[0], label='g', index=index, output_folder=input_folder)
            os.remove(frame0_info[1])
        elif read_key == ord('s'):
            persistence.save_label(frame=frame0_info[0], label='s', index=index, output_folder=input_folder)
            os.remove(frame0_info[1])


manual_review(DEFAULT_INPUT, input_folder=os.path.splitext(DEFAULT_INPUT)[0])