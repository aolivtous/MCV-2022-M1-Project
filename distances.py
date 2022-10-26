from pydoc import describe
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
    weights = global_variables.weights
    dists = {}
    idx_1 = idx_2 = idx_3 = idx_gray = []
    hist_ch1_db = hist_ch2_db = hist_ch3_db = np.array([])
    to_delete = False

    # NaN deletion for color histograms
    if weights["color"] and (np.isnan(hist_image.hist_ch1).any() or np.isnan(hist_image.hist_ch2).any() or np.isnan(hist_image.hist_ch3).any()):
        idx_1 = np.argwhere(np.isnan(hist_image.hist_ch1))
        idx_2 = np.argwhere(np.isnan(hist_image.hist_ch2))
        idx_3 = np.argwhere(np.isnan(hist_image.hist_ch3))

        hist_image.hist_ch1= np.delete(hist_image.hist_ch1, idx_1)
        hist_image.hist_ch2= np.delete(hist_image.hist_ch2, idx_2)
        hist_image.hist_ch3= np.delete(hist_image.hist_ch3, idx_3)

        to_delete = True

    # NaN deletion for texture coefficients
    elif weights["texture"] and np.isnan(hist_image.coeffs_dct).any():
        idx_gray = np.argwhere(np.isnan(hist_image.coeffs_dct))
            
        hist_image.coeffs_dct = np.delete(hist_image.coeffs_dct, idx_gray)

        to_delete = True
       
    for key_db, img_db in db_descriptors.items():
        if weights["color"]:
            hist_ch1_db = np.array(img_db.hist_ch1)
            hist_ch2_db = np.array(img_db.hist_ch2)
            hist_ch3_db = np.array(img_db.hist_ch3)
            if to_delete:
                hist_ch1_db = np.delete(hist_ch1_db, idx_1)
                hist_ch2_db = np.delete(hist_ch2_db, idx_2)
                hist_ch3_db = np.delete(hist_ch3_db, idx_3)
        
        if weights["texture"]:
            hist_lbp_db = np.array(img_db.coeffs_dct)
            if to_delete:
                hist_lbp_db = np.delete(hist_lbp_db, idx_gray)

        dist_color = dist_texture = dist_text = 0
        if weights["color"]:
            dist_ch1 = cv2.compareHist(hist_image.hist_ch1, hist_ch1_db, global_variables.argument_relations[distance_type])
            dist_ch2 = cv2.compareHist(hist_image.hist_ch2, hist_ch2_db, global_variables.argument_relations[distance_type])
            dist_ch3 = cv2.compareHist(hist_image.hist_ch3, hist_ch3_db, global_variables.argument_relations[distance_type])
            dist_color = np.mean(np.array([dist_ch1, dist_ch2, dist_ch3]))
        if weights["texture"]:
            dist_texture = np.linalg.norm(hist_image.coeffs_dct - img_db.coeffs_dct)
        # if weight_text > 0:

        dist = dist_color*weights["color"] + dist_texture*weights["texture"]
        dists[key_db] = distances(dist)

    return dists
