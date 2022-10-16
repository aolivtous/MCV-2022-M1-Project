from main import *
import main


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

def query_measures_colour(hist_query, hist_bbdd, distance_type):
    """
    It calculates the distance between the histograms of the query images and the database images
    
    :param hist_query: a dictionary of histograms of the query images
    :param hist_bbdd: a dictionary of histograms of the database images
    :param distance_type: the type of distance to use
    :return: A dictionary of dictionaries. The first level has the keys of the query images. The
    second level has the keys of the database images. The values of the second dictionary are the
    distances between the query and database images.
    """
    
    dists = {}
    for key_query, img_query in hist_query.items():
        print('key', key_query)

        dists[key_query]={}
        idx_1 = idx_2 = idx_3 = []
        to_delete = False

        if np.isnan(img_query.hist_ch1).any() or np.isnan(img_query.hist_ch2).any() or np.isnan(img_query.hist_ch3).any():
            idx_1 = np.argwhere(np.isnan(img_query.hist_ch1))
            idx_2 = np.argwhere(np.isnan(img_query.hist_ch2))
            idx_3 = np.argwhere(np.isnan(img_query.hist_ch3))

            img_query.hist_ch1= np.delete(img_query.hist_ch1, idx_1)
            img_query.hist_ch2= np.delete(img_query.hist_ch2, idx_2)
            img_query.hist_ch3= np.delete(img_query.hist_ch3, idx_3)
            to_delete = True

        for key_bbdd, img_bbdd in hist_bbdd.items():
            
            hist_ch1_bbdd = np.array(img_bbdd.hist_ch1)
            hist_ch2_bbdd = np.array(img_bbdd.hist_ch2)
            hist_ch3_bbdd = np.array(img_bbdd.hist_ch3)

            if to_delete:
                hist_ch1_bbdd = np.delete(hist_ch1_bbdd, idx_1)
                hist_ch2_bbdd = np.delete(hist_ch2_bbdd, idx_2)
                hist_ch3_bbdd = np.delete(hist_ch3_bbdd, idx_3)

            if distance_type == 'eucli':
                dist_ch1 = cv2.norm(img_query.hist_ch1, hist_ch1_bbdd, normType=cv2.NORM_L2)
                dist_ch2 = cv2.norm(img_query.hist_ch2, hist_ch2_bbdd, normType=cv2.NORM_L2)
                dist_ch3 = cv2.norm(img_query.hist_ch3, hist_ch3_bbdd, normType=cv2.NORM_L2)
                dist = np.mean(np.array([dist_ch1, dist_ch2, dist_ch3]))
            else:
                dist_ch1 = cv2.compareHist(img_query.hist_ch1, hist_ch1_bbdd, main.argument_relations[distance_type])
                dist_ch2 = cv2.compareHist(img_query.hist_ch2, hist_ch2_bbdd, main.argument_relations[distance_type])
                dist_ch3 = cv2.compareHist(img_query.hist_ch3, hist_ch3_bbdd, main.argument_relations[distance_type])
                dist = np.mean(np.array([dist_ch1, dist_ch2, dist_ch3]))
            dists[key_query][key_bbdd] = distances(dist)

    return dists

def query_measures_colour_3D(hist_query, hist_bbdd, distance_type):
    """
    It calculates the distance between the histograms of the query images and the database images
    
    :param hist_query: a dictionary of histograms of the query images
    :param hist_bbdd: a dictionary of histograms of the database images
    :param distance_type: the type of distance to use
    :return: A dictionary of dictionaries. The first level has the keys of the query images. The
    second level has the keys of the database images. The values of the second dictionary are the
    distances between the query and database images.
    """
    

    dists = {}
    for key_query, img_query in hist_query.items():
        dists[key_query]={}
        for key_bbdd, img_bbdd in hist_bbdd.items():

            #agafem ch1 pero en realitat en aquell argument hi ha guardat el histograma 3d dels tres canals ( no he canviat la classe histogram, que tenia 3 subhistogrames)
            if distance_type == 'eucli':
                dist = cv2.norm(img_query.hist_ch1, img_bbdd.hist_ch1, normType=cv2.NORM_L2)

            else:
                dist = cv2.compareHist(img_query.hist_ch1, img_bbdd.hist_ch1, main.argument_relations[distance_type])
                
            dists[key_query][key_bbdd] = distances(dist)

    return dists

def get_sorted_list_of_lists(dists, distance_type):
    """
    It takes a dictionary of dictionaries and returns a list of lists, where each list is a sorted list
    of the keys of the inner dictionaries
    
    :param dists: a dictionary of dictionaries, where the key of the outer dictionary is the query
    image, and the key of the inner dictionary is the database image. The value of the inner dictionary
    is an object of type Distance, which contains the distance between the query image and the database
    image
    :param distance_type: the type of distance metric to use.
    :return: A list of lists, where each list is the sorted list of image index results for a query.
    """
    list_of_lists = []
    if distance_type == "eucli" or distance_type ==  "hellin" or distance_type == "chisq":
        reverse = False
    else:
        reverse = True
    
    # Sort dict to assure the order
    dists = dict(sorted(dists.items(),key=lambda x:x[0]))
    for key_query, _ in dists.items():
        sorted_list = [int(key) for key, _ in sorted(dists[key_query].items(), key=lambda item: item[1].dist, reverse=reverse)]
        list_of_lists.append(sorted_list)

    return list_of_lists
