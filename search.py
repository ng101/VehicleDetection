import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import feature_extraction
import heat

# Define a single function that can extract features using hog sub-sampling and make predictions

def sample_hog(hogs, ypos, xpos, ncells_per_window, channel='ALL'):
    if channel == 'ALL':
        hog_features = []
        for i in range(len(hogs)):
            hog_features.append(hogs[i][ypos:ypos+ncells_per_window, xpos:xpos+ncells_per_window])
        hog_features = np.ravel(hog_features)
    else:
        hog_features = hogs[channel][ypos:ypos+ncells_per_window, xpos:xpos+ncells_per_window].ravel()

    return hog_features

 
def find_cars(img_vec, feature_params, window_size, 
        ystart, ystop, xstart, xstop, yscale, xscale,
        cells_per_step, clf, scaler):
    img_tosearch = feature_extraction.convert_color(img_vec,
            color_space=feature_params['colorspace'])
    img_tosearch = feature_extraction.normalize_255(img_tosearch)
    img_tosearch = img_tosearch[ystart:ystop, xstart:xstop,:]

    if xscale != 1 or yscale != 1:
        imshape = img_tosearch.shape
        img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/xscale), np.int(imshape[0]/yscale)))

    ch1 = img_tosearch[:,:,0]
    ch2 = img_tosearch[:,:,1]
    ch3 = img_tosearch[:,:,2]

    # Define blocks and steps as above
    pix_per_cell = feature_params['pixels_per_cell']
    nxcells = (ch1.shape[1] // pix_per_cell) - 1
    nycells = (ch1.shape[0] // pix_per_cell) - 1 

    cell_per_block = feature_params['cells_per_block']
    orient = feature_params['orient']

    ncells_per_window = (window_size // pix_per_cell) - 1 
    nxsteps = (nxcells - ncells_per_window) // cells_per_step
    nysteps = (nycells - ncells_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = feature_extraction.get_hog_features(ch1, orient, pix_per_cell,
            cell_per_block, feature_vec=False)
    hog2 = feature_extraction.get_hog_features(ch2, orient, pix_per_cell,
            cell_per_block, feature_vec=False)
    hog3 = feature_extraction.get_hog_features(ch3, orient, pix_per_cell,
            cell_per_block, feature_vec=False)

    bboxes = []
    allbboxes = []
    prob = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            if feature_params['hog_feat'] is True:
                hog_features = sample_hog([hog1, hog2, hog3],
                        ypos, xpos, ncells_per_window,
                        feature_params['hog_channel'])
            else:
                hog_features = np.array([])

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(img_tosearch[ytop:ytop+window_size,
                xleft:xleft+window_size], (64,64))

            # Extract other features
            spatial_size = feature_params['spatial_size']
            hist_bins = feature_params['hist_bins']
            spatial_feat = feature_params['spatial_feat']
            hist_feat = feature_params['hist_feat']

            other_features = feature_extraction.extract_features_from_image_vec(
                    subimg, spatial_size, hist_bins, spatial_feat=spatial_feat,
                    hist_feat=hist_feat, hog_feat=False)
            all_features = other_features + [hog_features]
            comb_features = np.concatenate(all_features).reshape(1, -1)
            # Scale features and make a prediction
            test_features = scaler.transform(comb_features)
            test_prediction = clf.predict(test_features)

            xbox_left = np.int(xleft * xscale)
            ytop_draw = np.int(ytop * yscale)
            xwin = np.int(window_size * xscale)    
            ywin = np.int(window_size * yscale)    
            bbox = ((xbox_left + xstart, ytop_draw + ystart), 
                    (xbox_left + xstart + xwin, ytop_draw + ywin + ystart))
            allbboxes.append(bbox) 

            if test_prediction == 1:
                decision_func = clf.decision_function(test_features)
                bboxes.append(bbox)
                prob.append(decision_func)

    return bboxes, prob, allbboxes

def search(img_vec, clf, scaler, feature_params, window_size, heat_map=None):
    if heat_map is None:
        heat_map = np.zeros_like(img_vec[:, :, 0]).astype(np.float)
    y_start_stops = [(400, 425), (400, 700), (400, 700)]
    x_start_stops = [(500, 900), (200, 1260), (200, 1260)]
    yscales = [0.25, 1.5, 1.5]
    xscales = [0.25, 1.5, 2]
    cells_per_step = [4, 2, 1]
    bboxes = []
    allbboxes = []
    prob = []
    #for i in range(len(scales)):
    for i in range(len(yscales)):
        ystart, ystop = y_start_stops[i][0], y_start_stops[i][1]
        xstart, xstop = x_start_stops[i][0], x_start_stops[i][1]
        b, p, a = find_cars(img_vec, feature_params, window_size,
                ystart, ystop, xstart, xstop, yscales[i], xscales[i],
                cells_per_step[i], clf, scaler)
        
        bboxes += b
        prob += p
        allbboxes += a
    heat_map = heat.add_heat(heat_map, bboxes, prob)
    return bboxes, prob, heat_map, allbboxes
