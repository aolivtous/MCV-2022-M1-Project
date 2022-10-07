"""
    Main file of the project
"""

# External modules
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.ma.core import mean
import pickle

# Internal modules
import histograms
import distances
import scores

# Code argument
name_bbdd = 'BBDD'
name_query = 'qsd1_w1'
color_code = "LAB" # ["RGB", "HSV", "LAB", "YCrCb"]
distance_type = 'hellin' # Possible arguments of distance_type at argument_relations
argument_relations = { 
    'corr': cv2.HISTCMP_CORREL,
    'chisq': cv2.HISTCMP_CHISQR,
    'intersect': cv2.HISTCMP_INTERSECT,
    'hellin': cv2.HISTCMP_BHATTACHARYYA,
    'eucli': False
}

# Global variable
base_dir = "C:/Users/marco/Desktop/master/introduction cv/project/"

query1_solutions = {}
with open( base_dir + 'qsd1_w1/gt_corresps.pkl', "rb" ) as f:
	query1_solutions = pickle.load(f)

def main():
    print('Main execution')
    # Assign directory
    directory_bbdd = base_dir + name_bbdd
    directory_query = base_dir + name_query

    # Generating DB and query dictionary of histograms
    hist_query = histograms.get_histograms(directory_query, color_code, False)
    hist_bbdd = histograms.get_histograms(directory_bbdd, color_code, True)

    # Calculating distances between the histograms
    dists = distances.query_measures_colour(hist_query, hist_bbdd, distance_type)

    dict_sorted= distances.get_sorted_dict(dists,distance_type)
    list_sorted=[]
    for key_dict,item_dict in dict_sorted.items():
        list_sorted.append(dict_sorted[key_dict])

    for i in range(len(list_sorted)):
        list_sorted[i][0] = int(list_sorted[i][0])

    mapk5 = scores.mapk(query1_solutions,list_sorted,5)
    print(mapk5)


if __name__ == "__main__":
    main()
