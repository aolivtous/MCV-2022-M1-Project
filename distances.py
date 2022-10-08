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
    dists = {}
    for key_query, img_query in hist_query.items():
        dists[key_query]={}
        for key_bbdd, img_bbdd in hist_bbdd.items():
            if distance_type == 'eucli':
                dist_ch1 = cv2.norm(img_query.hist_ch1, img_bbdd.hist_ch1, normType=cv2.NORM_L2)
                dist_ch2 = cv2.norm(img_query.hist_ch2, img_bbdd.hist_ch2, normType=cv2.NORM_L2)
                dist_ch3 = cv2.norm(img_query.hist_ch3, img_bbdd.hist_ch3, normType=cv2.NORM_L2)
                dist = np.mean(np.array([dist_ch1, dist_ch2, dist_ch3]))
            else:
                dist_ch1 = cv2.compareHist(img_query.hist_ch1, img_bbdd.hist_ch1, main.argument_relations[distance_type])
                dist_ch2 = cv2.compareHist(img_query.hist_ch2, img_bbdd.hist_ch2, main.argument_relations[distance_type])
                dist_ch3 = cv2.compareHist(img_query.hist_ch3, img_bbdd.hist_ch3, main.argument_relations[distance_type])
                dist = np.mean(np.array([dist_ch1, dist_ch2, dist_ch3]))
            dists[key_query][key_bbdd] = distances(dist)

    
    return dists



def get_sorted_dict(dists,distance_type):
    for key_query, img_query in dists.items():
        #sort_by_value = dict(sorted(dists[key_query].items(), key=lambda item: item[1].dist)[:5])
        if distance_type == "eucli" or "hellin" or "chisq":
            sorted_list = sorted(dists[key_query].items(), key=lambda item: item[1].dist, reverse =False)
        else:
            sorted_list = sorted(dists[key_query].items(), key=lambda item: item[1].dist, reverse =True)
        for i in range(len(sorted_list)):
            sorted_list[i] = sorted_list[i][0]
        dists[key_query] = sorted_list

    return dists
