from pydoc import describe
import global_variables
import cv2
import numpy as np
import textdistance

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

def query_measures(hist_image, db_descriptors, distance_type,  text):
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
    idx_1 = idx_2 = idx_3 = idx_dct = []
    hist_ch1_db = hist_ch2_db = hist_ch3_db = coeffs_dct_db = np.array([])
    to_delete = False

    # NaN deletion for color histograms
    if weights["color"] and (np.isnan(hist_image.hist_ch1).any() or np.isnan(hist_image.hist_ch2).any() or np.isnan(hist_image.hist_ch3).any()):
        print('Found NaN in hist -------------')
        idx_1 = np.argwhere(np.isnan(hist_image.hist_ch1))
        idx_2 = np.argwhere(np.isnan(hist_image.hist_ch2))
        idx_3 = np.argwhere(np.isnan(hist_image.hist_ch3))

        hist_image.hist_ch1= np.delete(hist_image.hist_ch1, idx_1)
        hist_image.hist_ch2= np.delete(hist_image.hist_ch2, idx_2)
        hist_image.hist_ch3= np.delete(hist_image.hist_ch3, idx_3)

        to_delete = True

    # NaN deletion for texture coefficients
    elif weights["texture"] and np.isnan(hist_image.coeffs_dct).any():
        idx_dct = np.argwhere(np.isnan(hist_image.coeffs_dct))
        
        hist_image.coeffs_dct = np.delete(hist_image.coeffs_dct, idx_dct)

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
            
            coeffs_dct_db = np.array(img_db.coeffs_dct)
            if to_delete:
                coeffs_dct_db = np.delete(coeffs_dct_db, idx_dct)


        dist_color = dist_texture = dist_text = 0

        if weights["color"]: #hellinger distance --> 0 distance --> most similarity
            dist_ch1 = cv2.compareHist(hist_image.hist_ch1, hist_ch1_db, global_variables.argument_relations[distance_type])
            dist_ch2 = cv2.compareHist(hist_image.hist_ch2, hist_ch2_db, global_variables.argument_relations[distance_type])
            dist_ch3 = cv2.compareHist(hist_image.hist_ch3, hist_ch3_db, global_variables.argument_relations[distance_type])
            dist_color = np.mean(np.array([dist_ch1, dist_ch2, dist_ch3]))
            if(dist_color > 1):
                print("dist_color > 1")
                print(dist_color)

        if weights["texture"]:
            # MinMax normalisation of the coefficients before calculating the distance
            # norm_hist_image_coeffs_dct = (hist_image.coeffs_dct - hist_image.coeffs_dct.min()) / (hist_image.coeffs_dct.max() - hist_image.coeffs_dct.min())
            # norm_coeffs_dct_db = (coeffs_dct_db - coeffs_dct_db.min()) / (coeffs_dct_db.max() - coeffs_dct_db.min())
            # Standard normalisation of the coefficients before calculating the distance
            norm_hist_image_coeffs_dct = (hist_image.coeffs_dct - hist_image.coeffs_dct.mean()) / hist_image.coeffs_dct.std()
            norm_coeffs_dct_db = (coeffs_dct_db - coeffs_dct_db.mean()) / coeffs_dct_db.std()
            dist_texture = np.linalg.norm(norm_hist_image_coeffs_dct - norm_coeffs_dct_db)
            if(dist_texture > 1):
                print("dist_texture > 1")
                print(dist_texture)

        if weights['color'] and weights['texture']:
            concatenation_features_query = np.float32(np.array([]))
            concatenation_features_query = np.append(concatenation_features_query, hist_image.hist_ch1)
            concatenation_features_query = np.append(concatenation_features_query, hist_image.hist_ch2)
            concatenation_features_query = np.append(concatenation_features_query, hist_image.hist_ch3)
            concatenation_features_query = np.append(concatenation_features_query, norm_hist_image_coeffs_dct)

            concatenation_features_bd = np.float32(np.array([]))
            concatenation_features_bd = np.append(concatenation_features_bd, hist_ch1_db)
            concatenation_features_bd = np.append(concatenation_features_bd, hist_ch2_db)
            concatenation_features_bd = np.append(concatenation_features_bd, hist_ch3_db)
            concatenation_features_bd = np.append(concatenation_features_bd, norm_coeffs_dct_db)
            #concatenation_features = np.append(concatenation_features, hist_ch2)
            #concatenation_features = np.append(concatenation_features, hist_ch3)
            dist_concat = np.linalg.norm(concatenation_features_query - concatenation_features_bd )


            
        if weights["text"]:
            
            with open(global_variables.dir_db + 'bbdd_' + key_db + '.txt') as f:
                first_line = f.readline()

            db_text =  first_line.split(",")[0].replace("'","").replace("(","")

            dist_text = textdistance.levenshtein.normalized_distance(text, db_text)

        #dist = dist_color*dist_texture

        #dist = dits_color*weights["color"] + dist_texture*weights["texture"]+dist_text*weights["text"]
        dist = dist_concat

        dists[key_db] = distances(dist)

    return dists

