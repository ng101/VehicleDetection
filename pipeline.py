from moviepy.editor import VideoFileClip
import pickle
import matplotlib.pyplot as plt
import cv2 
import numpy as np
import sys
import feature_extraction
import draw_boxes
import heat
import search

with open("classifiers/classify.p", "rb") as f:
    clf_params = pickle.load(f)

clf = clf_params["clf"]
scaler = clf_params["scl"]
feature_params = clf_params['params']
window_size = clf_params['window_shape'][0]

def process_image(img_vec):
    bboxes, hmap, labels = search.search(img_vec, clf, scaler,
                                         feature_params, window_size)
    new_img_vec = heat.draw_labeled_bboxes(img_vec, labels)
    return new_img_vec 

PREFIX = '../'

if __name__ == '__main__':
    output = 'project_video.mp4'
    if len(sys.argv) >= 2:
        output = sys.argv[1]

    mode = 'prod'
    if len(sys.argv) >= 3:
        mode = sys.argv[2]

    clip = VideoFileClip(PREFIX + output)
    if mode == 'debug':
        out_clip = clip.fl_image(debug_image)
    else:
        out_clip = clip.fl_image(process_image)
    out_clip.write_videofile(output, audio=False)
