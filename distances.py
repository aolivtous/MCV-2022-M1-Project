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

def query_measures_colour(hist_image, db_descriptors, distance_type, descriptors):
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


    if len(descriptors) == 1 and "color" in descriptors and (np.isnan(hist_image.hist_ch1).any() or np.isnan(hist_image.hist_ch2).any() or np.isnan(hist_image.hist_ch3).any()):
        idx_1 = np.argwhere(np.isnan(hist_image.hist_ch1))
        idx_2 = np.argwhere(np.isnan(hist_image.hist_ch2))
        idx_3 = np.argwhere(np.isnan(hist_image.hist_ch3))

        hist_image.hist_ch1= np.delete(hist_image.hist_ch1, idx_1)
        hist_image.hist_ch2= np.delete(hist_image.hist_ch2, idx_2)
        hist_image.hist_ch3= np.delete(hist_image.hist_ch3, idx_3)

        to_delete = True

    elif len(descriptors) == 1 and "texture" in descriptors and np.isnan(hist_image.coeffs_dct).any():
        idx_gray = np.argwhere(np.isnan(hist_image.coeffs_dct))
            
        hist_image.coeffs_dct = np.delete(hist_image.coeffs_dct, idx_gray)

        to_delete = True

    elif len(descriptors) == 2 and (np.isnan(hist_image.hist_ch1).any() or np.isnan(hist_image.hist_ch2).any() or np.isnan(hist_image.hist_ch3).any() or np.isnan(hist_image.coeffs_dct).any()):
        idx_gray = np.argwhere(np.isnan(hist_image.coeffs_dct)) 
        hist_image.coeffs_dct = np.delete(hist_image.coeffs_dct, idx_gray)

        idx_1 = np.argwhere(np.isnan(hist_image.hist_ch1))
        idx_2 = np.argwhere(np.isnan(hist_image.hist_ch2))
        idx_3 = np.argwhere(np.isnan(hist_image.hist_ch3))

        hist_image.hist_ch1= np.delete(hist_image.hist_ch1, idx_1)
        hist_image.hist_ch2= np.delete(hist_image.hist_ch2, idx_2)
        hist_image.hist_ch3= np.delete(hist_image.hist_ch3, idx_3)

        to_delete = True
       
    for key_db, img_db in db_descriptors.items():
        
        if 'color' in descriptors:
            hist_ch1_db = np.array(img_db.hist_ch1)
            hist_ch2_db = np.array(img_db.hist_ch2)
            hist_ch3_db = np.array(img_db.hist_ch3)
            if to_delete:
                hist_ch1_db = np.delete(hist_ch1_db, idx_1)
                hist_ch2_db = np.delete(hist_ch2_db, idx_2)
                hist_ch3_db = np.delete(hist_ch3_db, idx_3)
        
        if 'texture' in descriptors:
            hist_lbp_db = np.array(img_db.coeffs_dct)
            if to_delete:
                hist_lbp_db = np.delete(hist_lbp_db, idx_gray)


        if len(descriptors) == 1 and "color" in descriptors:
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
        
        elif len(descriptors) == 1 and "texture" in descriptors:
            '''if distance_type == 'eucli':
                dist_gray = cv2.norm(hist_image.hist_lbp, hist_lbp_db, normType=cv2.NORM_L2)
            else:
                dist_gray = cv2.compareHist(hist_image.hist_lbp, hist_lbp_db, global_variables.argument_relations[distance_type])'''

            dists[key_db] = distances(np.linalg.norm(hist_image.coeffs_dct-img_db.coeffs_dct))#distances(dist_gray))

        else:
            if distance_type == 'eucli':
                '''concatenation_array_image = np.concatenate([hist_image.coeffs_dct, hist_image.hist_ch1,hist_image.hist_ch2,hist_image.hist_ch3])
                concatenation_array_db = np.concatenate([hist_lbp_db,hist_ch1_db,hist_ch2_db,hist_ch3_db])

                dist = np.linalg.norm(concatenation_array_image-concatenation_array_db)'''

                #dist = cv2.compareHist(concatenation_array_image, concatenation_array_db, global_variables.argument_relations[distance_type])
                #dist_gray = cv2.norm(hist_image.hist_lbp, hist_lbp_db, normType=cv2.NORM_L2)
                weight_texture = 0.5
                weight_color = 0.5
                dist_texture = np.linalg.norm(hist_image.coeffs_dct-img_db.coeffs_dct)
                dist_ch1 = cv2.norm(hist_image.hist_ch1, hist_ch1_db, normType=cv2.NORM_L2)
                dist_ch2 = cv2.norm(hist_image.hist_ch2, hist_ch2_db, normType=cv2.NORM_L2)
                dist_ch3 = cv2.norm(hist_image.hist_ch3, hist_ch3_db, normType=cv2.NORM_L2)
                dist_color = np.mean(np.array([dist_ch1, dist_ch2, dist_ch3]))
                dist = dist_texture*weight_texture + dist_color*weight_color
            else:
                
                '''dist_gray = cv2.compareHist(hist_image.hist_lbp, hist_lbp_db, global_variables.argument_relations[distance_type])
                dist_ch1 = cv2.compareHist(hist_image.hist_ch1, hist_ch1_db, global_variables.argument_relations[distance_type])
                dist_ch2 = cv2.compareHist(hist_image.hist_ch2, hist_ch2_db, global_variables.argument_relations[distance_type])
                dist_ch3 = cv2.compareHist(hist_image.hist_ch3, hist_ch3_db, global_variables.argument_relations[distance_type])
                dist = np.mean(np.array([dist_gray,dist_ch1, dist_ch2, dist_ch3]))'''
            dists[key_db] = distances(dist)

    return dists
