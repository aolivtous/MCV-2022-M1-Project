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

    dists_color, dists_texture, dists_text, match_feature = {}, {}, {}, {}
    dists_db = {}

    idx_1 = idx_2 = idx_3 = idx_dct = []
    hist_ch1_db = hist_ch2_db = hist_ch3_db = coeffs_dct_db = np.array([])
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
    if weights["texture"] and np.isnan(hist_image.coeffs_dct).any():
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

        if weights["color"]: #hellinger distance --> 0 distance --> most similarity
            dist_ch1 = cv2.compareHist(hist_image.hist_ch1, hist_ch1_db, global_variables.argument_relations[distance_type])
            dist_ch2 = cv2.compareHist(hist_image.hist_ch2, hist_ch2_db, global_variables.argument_relations[distance_type])
            dist_ch3 = cv2.compareHist(hist_image.hist_ch3, hist_ch3_db, global_variables.argument_relations[distance_type])
            dists_color[key_db] = np.mean(np.array([dist_ch1, dist_ch2, dist_ch3]))
        else:
            dists_color[key_db] = 0

        if weights["texture"]:
            # MinMax normalisation of the coefficients before calculating the distance
            # norm_hist_image_coeffs_dct = (hist_image.coeffs_dct - hist_image.coeffs_dct.min()) / (hist_image.coeffs_dct.max() - hist_image.coeffs_dct.min())
            # norm_coeffs_dct_db = (coeffs_dct_db - coeffs_dct_db.min()) / (coeffs_dct_db.max() - coeffs_dct_db.min())
            # Standard normalisation of the coefficients before calculating the distance

            norm_hist_image_coeffs_dct = (hist_image.coeffs_dct - hist_image.coeffs_dct.mean()) / hist_image.coeffs_dct.std()
            norm_coeffs_dct_db = (coeffs_dct_db - coeffs_dct_db.mean()) / coeffs_dct_db.std()
            dists_texture[key_db] = np.linalg.norm(norm_hist_image_coeffs_dct - norm_coeffs_dct_db)
        else:
            dists_texture[key_db] = 0
            
        if weights["text"]:
            
            with open(global_variables.dir_db + 'bbdd_' + key_db + '.txt') as f:
                first_line = f.readline()

            db_text =  first_line.split(",")[0].replace("'","").replace("(","")

            #for qsd1_w2
            """if(len(db_text)>1):
                db_text =  first_line.split(",")[1].replace("'","").replace(")","").replace("\n","")
            else :
                db_text = ""
            print(db_text)"""
            

            # Clean db_text and text to remove special characters
            # db_text = re.sub(r'[^\w\s]','',db_text) # Now this is done at main
            # text = re.sub(r'[^\w\s]','',text)
            text = text.lower()
            db_text = db_text.lower()

            dists_text[key_db] = textdistance.levenshtein.normalized_distance(text, db_text)
        else:
            dists_text[key_db] = 0

        if weights["feature"]:
            # query_feature_kp = cv2.KeyPoint(x=hist_image.feature[0][0],y=hist_image.feature[0][1],_size=hist_image.feature[1], _angle=hist_image.feature[2], _response=hist_image.feature[3], _octave=hist_image.feature[4], _class_id=hist_image.feature[5]) 
            query_feature_des = hist_image.feature
            # db_feature_kp = cv2.KeyPoint(x=img_db.feature[0][0],y=img_db.feature[0][1],_size=img_db.feature[1], _angle=img_db.feature[2], _response=img_db.feature[3], _octave=img_db.feature[4], _class_id=img_db.feature[5]) 
            db_feature_des = img_db.feature
            # print(f'num_matches for {key_db} = {db_feature_des}')

            INFINITY = 1e9
            if db_feature_des is None:
                dists_db[key_db] = distances(INFINITY)
            else:
                match_dist = match_features(query_feature_des, db_feature_des)
                # Check if nan dists_db[key_db] = distances(INFINITY)
                if np.isnan(match_dist):
                    dists_db[key_db] = distances(INFINITY)
                else:
                    dists_db[key_db] = distances(match_dist)

            print(f'Distance based on matches for {key_db} = {round(dists_db[key_db].dist, 3)}')


    if weights["feature"]:
        return dists_db
        # Refining of the feature distance
        # if weights["feature"]:
        #     # Invert the matches using max-value (in order to have "distance" meaning)
        #     dists_db = { k: distances(np.max(np.array(list(match_feature.values())))-v) for k, v in match_feature.items() }
    
    else:
        # Normalize values of dists_texture euclidean distances
        if weights["texture"]:
            dists_texture = { k: (v - np.mean(np.array(list(dists_texture.values())))) / np.std(np.array(list(dists_texture.values()))) for k, v in dists_texture.items() }

        # Normalization of the euclidean distance
        for key_db, img_db in db_descriptors.items():
            dists_db[key_db] = distances(weights["color"] * dists_color[key_db] + weights["texture"] * dists_texture[key_db] + weights["text"] * dists_text[key_db])
        
        return dists_db

def match_features(des1, des2):#, kp1=None, kp2=None):
    matches = []
    if global_variables.methods_search['default']['feature_algorithm'] == 'ORB' or global_variables.methods_search['default']['feature_algorithm'] == 'BRIEF':
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Matches is a list of DMatch objects that contains information about the keypoints matches (distance, queryIdx, trainIdx, imgIdx)
        # Sort them in the order of their distance.
        # matches = sorted(matches, key = lambda x:x.distance)
        # num_matches = len(matches)
        # # ! Play with the distance threshold to get better results
        # # Get the first 10% of the matches
        # top_num_matches = int(len(matches)*0.1)
        # # Draw first 10 matches.
        # top_matches = matches[:top_num_matches]
        # # Get the distance attribute of the top_matches
        # top_matches_dist_mean = np.mean([m.distance for m in top_matches])
        # return top_matches_dist_mean

    elif global_variables.methods_search['default']['match_algorithm'] == 'BF':
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        matches = good

    elif global_variables.methods_search['default']['match_algorithm'] == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        matches = good
    else:
        print('Match algorithm not found')

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    num_matches = len(matches)
    # ! Play with the distance threshold to get better results
    # Get the first 10% of the matches
    top_num_matches = int(len(matches)*0.1)
    # Draw first 10 matches.
    top_matches = matches[:top_num_matches]
    # Get the distance attribute of the top_matches
    top_matches_dist_mean = np.mean([m.distance for m in top_matches])
    return top_matches_dist_mean

    

    
        

