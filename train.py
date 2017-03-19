import classifier
import feature_extraction
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import pickle
import datetime
import time
import matplotlib.image as  mpimg

# Params
params = {
        'colorspace': 'YCrCb', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        'spatial_size': (32, 32),
        'hist_bins': 32,
        'orient': 18,
        'pixels_per_cell': 8,
        'cells_per_block': 2,
        'hog_channel': 'ALL', # Can be 0, 1, 2, or "ALL"
        'spatial_feat': True,
        'hist_feat': True,
        'hog_feat': True
        }


# Divide up into cars and notcars
cars = glob.glob('../vehicles/*/*.png')
noncars = glob.glob('../non-vehicles/*/*.png')

# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    img = mpimg.imread(car_list[0])
    data_dict["image_shape"] = img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = img.dtype
    # Return data_dict
    return data_dict

look = data_look(cars, noncars)
print(look)
print(len(cars), len(noncars))

features = feature_extraction.extract_features_from_images(cars + noncars, 
        params['colorspace'], params['spatial_size'], params['hist_bins'],
        params['orient'], params['pixels_per_cell'], params['cells_per_block'],
        params['hog_channel'], params['spatial_feat'], 
        params['hist_feat'], params['hog_feat'])

X = np.vstack(features).astype(np.float64)                        
print(X.shape)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(cars)), np.zeros(len(noncars))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

# Check the training time for the SVC
t=time.time()
clf = classifier.get_linear_svc(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
accuracy = round(clf.score(X_test, y_test), 4)
print('Test Accuracy of SVC = ', accuracy)
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels')
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H:%M:%S')
with open('classifiers/classify-{}-{}.p'.format(accuracy, st), 'wb') as f:
    pickle.dump({'clf': clf, 'scl': X_scaler,
        'params': params, 'window_shape': look['image_shape'][0:2]},
        f)
