from main import *

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

def get_top5(dists):
    sort_by_value = dict(sorted(dists.items(), key=lambda item: item[1].dist)[:5])

    for key_dist, dist in sort_by_value.items():
        print(f'Sim intersection with {key_dist} is: {dist.dist}')
        print(f'Sim correlation  with {key_dist} is: {dist.sim_correlation_rgb}')
        print(f'RGB euclidean distance with {key_dist} is: {dist.dist_eucl_rgb}')
        print(f'RGB chi square distance with {key_dist} is: {dist.dist_chisq_rgb}')
        print(f'RGB Heilinger distance with {key_dist} is: {dist.dist_heilinger_rgb}')