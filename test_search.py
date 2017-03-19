import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import search
import glob
import feature_extraction
import draw_boxes
import heat

with open("classifiers/classify.p", "rb") as f:
    clf_params = pickle.load(f)

clf = clf_params["clf"]
scaler = clf_params["scl"]
feature_params = clf_params['params']
window_size = clf_params['window_shape'][0]

images = glob.glob('../test_images/test*.jpg')

for img_name in images:
    img_vec = mpimg.imread(img_name)
    bboxes, prob, hmap, allboxes = search.search(img_vec, clf, scaler,
                                         feature_params, window_size)
    labels = heat.get_labels(hmap)
    print(bboxes, prob)
    search_img_vec = draw_boxes.draw_boxes(img_vec, allboxes, (255, 0, 0))
    new_img_vec = heat.draw_labeled_bboxes(img_vec, labels)
    #plt.imshow(new_img_vec)
    #plt.show()
    print("{} cars found".format(labels[1]))
    f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(search_img_vec)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(new_img_vec)
    ax2.set_title('Output', fontsize=40)

    ax3.imshow(labels[0], cmap='gray')
    ax3.set_title('Heat map')

    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    f.savefig('output_images/out_{}'.format(img_name.split('/')[-1]))
    plt.close(f)

