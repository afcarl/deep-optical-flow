import cv2
import pickle
import os
import flow

FRAME_LABEL_FORMAT = "{:s}/{:05d}.{:s}.o.jpeg"
FRAME_LABEL_ANNOTATION = "{:s}/{:05d}.a.jpeg"
FRAME_FLOW_FORMAT = "{:s}/flow.{:05d}.{:s}.flo"


def save_label(frame, label, index, output_folder, dense_flow=None, annotate=False):
    frame_filename = FRAME_LABEL_FORMAT.format(output_folder, int(index), label)

    os.makedirs(os.path.dirname(frame_filename), exist_ok=True)

    cv2.imwrite(frame_filename, frame)

    if dense_flow is not None:
        if annotate:
            cv2.imwrite(FRAME_LABEL_ANNOTATION.format(output_folder, int(index)), flow.draw_flow(frame, dense_flow, step=8))
        with open(FRAME_FLOW_FORMAT.format(output_folder, int(index), label), "+wb") as flow_file:
            pickle.dump(dense_flow, flow_file)
