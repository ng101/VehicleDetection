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
import track_and_filter

with open("classifiers/classify.p", "rb") as f:
    clf_params = pickle.load(f)

clf = clf_params["clf"]
scaler = clf_params["scl"]
feature_params = clf_params['params']
window_size = clf_params['window_shape'][0]

tracker = track_and_filter.Track()

def process_image(img_vec):
    bboxes, prob, hmap, allboxes = search.search(img_vec, clf, scaler,
                                         feature_params, window_size)
    #labels = heat.get_labels(hmap)
    new_labels = tracker.track(img_vec, hmap)
    new_img_vec = heat.draw_labeled_bboxes(img_vec, new_labels)
    #chmap, centroids, filt_bboxes = tracker.track(img_vec, bboxes, hmap, labels)
    #new_img_vec = draw_boxes.draw_boxes(img_vec, filt_bboxes)
    #new_img_vec[chmap] = [255, 0, 0]
    #plt.imshow(new_img_vec)
    #plt.show()
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
