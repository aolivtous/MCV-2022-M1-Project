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

# Code argument
name_bbdd = 'BBDD'
name_query = 'qsd1_w1'
color_code = "LAB" # ["RGB", "HSV", "LAB", "YCrCb"]
distance_type = 'eucli' # Possible arguments of distance_type at argument_relations
argument_relations = { 
    'corr': cv2.HISTCMP_CORREL,
    'chisq': cv2.HISTCMP_CHISQR,
    'intersect': cv2.HISTCMP_INTERSECT,
    'hellin': cv2.HISTCMP_BHATTACHARYYA,
    'eucli': False
}

# Global variable
base_dir = "../"

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



if __name__ == "__main__":
    main()
