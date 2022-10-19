import global_variables
import cv2
import numpy as np

class distances: 
    instances = {}
    def __init__(self, dist): 
        self.dist = dist

# dists = {
#     '0000': {
#         '0000': {
#             'dist': 12
#         },
#         '0001': {
#             'dist': 15
#         }
#     }
# }

def query_measures_colour(hist_image, db_descriptors, distance_type):
    """
    It calculates the distance between the histograms of the query images and the database images
    
    :param hist_query: a dictionary of histograms of the query images
    :param hist_db: a dictionary of histograms of the database images
    :param distance_type: the type of distance to use
    :return: A dictionary of dictionaries. The first level has the keys of the query images. The
    second level has the keys of the database images. The values of the second dictionary are the
    distances between the query and database images.
    """
    
    dists = {}
    idx_1 = idx_2 = idx_3 = []
    to_delete = False

    if np.isnan(hist_image.hist_ch1).any() or np.isnan(hist_image.hist_ch2).any() or np.isnan(hist_image.hist_ch3).any():
        idx_1 = np.argwhere(np.isnan(hist_image.hist_ch1))
        idx_2 = np.argwhere(np.isnan(hist_image.hist_ch2))
        idx_3 = np.argwhere(np.isnan(hist_image.hist_ch3))

        hist_image.hist_ch1= np.delete(hist_image.hist_ch1, idx_1)
        hist_image.hist_ch2= np.delete(hist_image.hist_ch2, idx_2)
        hist_image.hist_ch3= np.delete(hist_image.hist_ch3, idx_3)
        to_delete = True

    for key_db, img_db in db_descriptors.items():
        
        hist_ch1_db = np.array(img_db.hist_ch1)
        hist_ch2_db = np.array(img_db.hist_ch2)
        hist_ch3_db = np.array(img_db.hist_ch3)

        if to_delete:
            hist_ch1_db = np.delete(hist_ch1_db, idx_1)
            hist_ch2_db = np.delete(hist_ch2_db, idx_2)
            hist_ch3_db = np.delete(hist_ch3_db, idx_3)

        if distance_type == 'eucli':
            dist_ch1 = cv2.norm(hist_image.hist_ch1, hist_ch1_db, normType=cv2.NORM_L2)
            dist_ch2 = cv2.norm(hist_image.hist_ch2, hist_ch2_db, normType=cv2.NORM_L2)
            dist_ch3 = cv2.norm(hist_image.hist_ch3, hist_ch3_db, normType=cv2.NORM_L2)
            dist = np.mean(np.array([dist_ch1, dist_ch2, dist_ch3]))
        else:
            dist_ch1 = cv2.compareHist(hist_image.hist_ch1, hist_ch1_db, global_variables.argument_relations[distance_type])
            dist_ch2 = cv2.compareHist(hist_image.hist_ch2, hist_ch2_db, global_variables.argument_relations[distance_type])
            dist_ch3 = cv2.compareHist(hist_image.hist_ch3, hist_ch3_db, global_variables.argument_relations[distance_type])
            dist = np.mean(np.array([dist_ch1, dist_ch2, dist_ch3]))
        dists[key_db] = distances(dist)

    return dists
