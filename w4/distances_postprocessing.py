from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import global_variables
import distances

def distances_postprocessing(dists):
    # dists is a dictionary with the distances of each painting in the query to each painting in the database
    # dists = {query_painting1: {db_painting1: distance, db_painting2: distance, ...}, query_painting2: {db_painting1: distance, db_painting2: distance, ...}, ...}
    db_dists_mean = {}
    for key_query, _ in dists.items():
        for key_db, dist in dists[key_query].items():
            if not key_db in db_dists_mean:
                db_dists_mean[key_db] = [dist.dist]
            else:
                db_dists_mean[key_db].append(dist.dist)
    
    for key_db, db_dists in db_dists_mean.items():
        db_dists_mean[key_db] = np.mean(db_dists)
    
    # Now normalize the distances
    for key_query, _ in dists.items():
        for key_db, dist in dists[key_query].items():
            dists[key_query][key_db] = distances.distances(dist.dist / db_dists_mean[key_db])
    return dists




