"""
    Main file of the project
"""

# External modules
import sys
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.ma.core import mean
import pickle
import statistics

# Internal modules
import utils
import histograms
import distances
import scores
import mask_v1
import mask_v2
import mask_evaluation

argument_relations = { 
    'corr': cv2.HISTCMP_CORREL,
    'chisq': cv2.HISTCMP_CHISQR,
    'intersect': cv2.HISTCMP_INTERSECT,
    'hellin': cv2.HISTCMP_BHATTACHARYYA,
    'eucli': False
}

methods_search = {
    1: {
        'color_code': 'LAB',
        'distance_type': 'hellin'
    },
    2: {
        'color_code': 'LAB',
        'distance_type': 'intersect'
    }
}

def main():
    """
    Main pipeline of the project.
    It generates masks for the query images, calculates the histograms for the query and database
    images, calculates the distances between the histograms, sorts the results and evaluates the
    algorithm.
    """
    # Default arguments
    name_db = 'db'
    name_query = 'qsd2_w1'
    method_search = 1
    color_code = "LAB" # ["RGB", "HSV", "LAB", "YCrCb"]
    distance_type = 'hellin' # Possible arguments of distance_type at argument_relations
    backgrounds = True
    solutions = True
    plot_histograms = False
    default_threshold = 95

    # Global variable
    base_dir = "../"
    output_name = "predictions"

    # If there are not enough arguments, exit the program.
    try:
        name_query = sys.argv[1]
        method_search = int(sys.argv[2])
        method_mask = int(sys.argv[3])
        backgrounds = bool(utils.str_to_bool(sys.argv[4]))
        solutions = bool(utils.str_to_bool(sys.argv[5]))
    except:
        print(f'Exiting. Not enough arguments ({len(sys.argv) - 1} of 5)')
        exit(1)

    # Directories assignment
    dir_db = base_dir + name_db
    dir_query = base_dir + name_query
    output_path = "/" + output_name
    directory_output = dir_query + output_path

    # Arguments bound checking
    if(method_search == 1 or method_search == 2):
        color_code = methods_search[method_search]['color_code']
        distance_type = methods_search[method_search]['distance_type']
    else:
        print('Exiting. Method search must be 1 or 2')
        exit(1)
    
    if(method_mask != 1 and method_mask != 2):
        print('Exiting. Method mask must be 1 or 2')
        exit(1)

    query_solutions = None
    if(solutions):
        try:
            with open( dir_query + '/gt_corresps.pkl', "rb" ) as f:
                query_solutions = pickle.load(f)
        except:
            pass
    
    try:
        os.makedirs(directory_output)
    except FileExistsError:
        # Directory already exists
        pass
    
    # Masks generation
    if(backgrounds):
        if(method_mask == 1):
            mask_v1.generate_masks(dir_query, directory_output, threshold_value = default_threshold, plot_histograms = plot_histograms)
        elif(method_mask == 2):
            mask_v2.generate_masks(dir_query, directory_output, plot_histograms = plot_histograms)

    # Generating DB and query dictionary of histograms
    hist_query = histograms.get_histograms(dir_query, output_name, color_code, query = True , with_mask = True and backgrounds)
    hist_db = histograms.get_histograms(dir_db, output_name, color_code, query = False , with_mask = False)

    # Calculating distances between the histograms
    dists = distances.query_measures_colour(hist_query, hist_db, distance_type)

    # Results sorting
    results_sorted = distances.get_sorted_list_of_lists(dists, distance_type)

    if(solutions):
        # Algorithm evaluation
        mapk1 = scores.mapk(query_solutions, results_sorted, k = 1)
        print(f'The map-1 score is: {round(mapk1, 2)}')
        mapk5 = scores.mapk(query_solutions, results_sorted, k = 5)
        print(f'The map-5 score is: {round(mapk5, 2)}')
        if(backgrounds):
            mask_evaluation.mask_eval_avg(directory_output, dir_query, print_each = False, print_avg = True)
    else:
        print('No solutions given --> evaluation not avaliable.')
    
    # Shorten of the results lists to k=10
    results_sorted_top = [l[:10] for l in results_sorted]
    for idx, l in enumerate(results_sorted_top):
        print(f'Result for image {idx}:', l)
    with open(directory_output + '/result.pkl', 'wb') as handle:
        pickle.dump(results_sorted_top, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
