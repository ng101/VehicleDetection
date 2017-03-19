import numpy as np
from scipy.ndimage.measurements import label
import heat

class Track:
    NUM_FRAMES_TO_TRACK = 5
    CENTROID_WINDOW_SIZE = 10  # CENTROID WINDOW SIZE
    def __init__(self):
        self.counter = 0
        self.hmaps = []
        self.chmaps = []

    def fits(self, point, paths):
        for p in paths:
            (x, y) = point
            est_x = p[0]*y**2 + p[1]*y + p[0]
            if np.abs(x - est_x) < 10:
                print('Fits return true')
                return True
        return False

    def hmap_of_centroids(self, labels, paths):
        c_hmap = np.zeros_like(labels[0])
        centroids = []
        bboxes = []
        for l in range(1, labels[1]+1):
            nonzero = (labels[0] == l).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            cx, cy = np.mean(nonzerox), np.mean(nonzeroy)
            ws = Track.CENTROID_WINDOW_SIZE
            c_hmap[cy-ws:cy+ws, cx-ws:cx+ws] = 1
            centroids.append((cx, cy))
            if (0 == len(paths)) or (self.fits((cx, cy), paths)):
                bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                        (np.max(nonzerox), np.max(nonzeroy)))
                bboxes.append(bbox)

        return c_hmap, centroids, bboxes

    def get_combined_chmap(self, img):
        chmaps_list = self.chmaps[-1 * Track.NUM_FRAMES_TO_TRACK:]
        chmap = np.zeros_like(img[:, :, 0])
        for c in chmaps_list:
            nonzero = c.nonzero()
            chmap[nonzero[0], nonzero[1]] = 1
        return chmap

    def get_paths(self, chmap):
        labels = label(chmap)
        paths = []
        for l in range(1, labels[1] + 1):
            nonzero = (labels[0] == l).nonzero()
            if len(nonzero[0]) > 5 * (2*Track.CENTROID_WINDOW_SIZE)**2:
                paths.append(np.polyfit(nonzero[0], nonzero[1], 2))
        return paths

    def track1(self, img, bboxes, hmap, labels):
        self.hmaps.append(hmap)
        integrated_hmap = sum(self.hmaps[-1*Track.NUM_FRAMES_TO_TRACK:])
        new_hmap = heat.apply_threshold(integrated_hmap, 2)
        new_labels = heat.get_labels(new_hmap)

    def track(self, img, bboxes, hmap, labels):
        self.counter += 1
        combined_hmap = self.get_combined_chmap(img)
        paths = self.get_paths(combined_hmap)
  
        # find new centroid hmap
        chmap, centroids, bboxes = self.hmap_of_centroids(labels, paths)
        self.chmaps.append(chmap)

        return combined_hmap, centroids, bboxes

