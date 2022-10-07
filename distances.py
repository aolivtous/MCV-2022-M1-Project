from main import *

class distances: 
    instances = {}
    def __init__(self, name, dist): 
        self.dist = dist
        distances.instances[name] = self

# dists = {}

def query_measures_colour(hist_query, hist_bbdd, distance_type):
    dists = {}
    
    for key_bbdd, img_bbdd in hist_bbdd.items():
        if distance_type == 'eucli':
            dist_ch1 = cv2.norm(hist_query.hist_ch1, img_bbdd.hist_ch1, normType=cv2.NORM_L2)
            dist_ch2 = cv2.norm(hist_query.hist_ch2, img_bbdd.hist_ch2, normType=cv2.NORM_L2)
            dist_ch3 = cv2.norm(hist_query.hist_ch3, img_bbdd.hist_ch3, normType=cv2.NORM_L2)
            dist = np.mean(np.array([dist_ch1, dist_ch2, dist_ch3]))
        else:
            dist_ch1 = cv2.compareHist(hist_query.hist_ch1, img_bbdd.hist_ch1, main.argument_relations[distance_type])
            dist_ch2 = cv2.compareHist(hist_query.hist_ch2, img_bbdd.hist_ch2, main.argument_relations[distance_type])
            dist_ch3 = cv2.compareHist(hist_query.hist_ch3, img_bbdd.hist_ch3, main.argument_relations[distance_type])
            dist = np.mean(np.array([dist_ch1, dist_ch2, dist_ch3]))

        dists[key_bbdd] = distances(key_bbdd, dist)

    return dists