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
import global_variables
import utils
import histograms
import distances
import scores
import masks
import mask_evaluation
import find_boxes

def main():
    """
    Main pipeline of the project.
    It generates masks for the query images, calculates the histograms for the query and database
    images, calculates the distances between the histograms, sorts the results and evaluates the
    algorithm.
    """
    ## Variables initialization

    # If there are not enough arguments, exit the program.
    try:
        name_query = sys.argv[1]
        method_search = int(sys.argv[2])
        backgrounds = bool(utils.str_to_bool(sys.argv[3]))
        has_boundingbox = bool(utils.str_to_bool(sys.argv[4]))
        may_have_split = bool(utils.str_to_bool(sys.argv[5]))
        solutions = bool(utils.str_to_bool(sys.argv[6]))
        recompute_db = bool(utils.str_to_bool(sys.argv[7]))
    except:
        print(f'Exiting. Not enough arguments ({len(sys.argv) - 1} of 6)')
        exit(1)

    global_variables.init(name_query)
    # Arguments bound checking
    if(method_search == 1 or method_search == 2):
        color_code = global_variables.methods_search[method_search]['color_code']
        distance_type = global_variables.methods_search[method_search]['distance_type']
    else:
        print('Exiting. Method search must be 1 or 2')
        exit(1)

    query_solutions = boxes_solutions = None
    if(solutions):
        try:
            with open(f'{global_variables.dir_query}gt_corresps.pkl', "rb" ) as f:
                query_solutions = pickle.load(f)
            if(has_boundingbox):
                with open(f'{global_variables.dir_query}text_boxes.pkl', 'rb') as f:
                    boxes_solutions = pickle.load(f)
        except:
            pass
    
    for dir in global_variables.new_dirs:
        try:
            os.makedirs(dir)
        except FileExistsError:
            # Directory already exists
            pass
    
    ### PIPELINE

    ## DB Descriptors extraction
    '''db_descriptors = {}
    if(recompute_db):
        print(f'Exctracting descriptors from DB directory: {global_variables.dir_db}')
        for filename in os.scandir(global_variables.dir_db):
            f = os.path.join(global_variables.dir_db, filename)
            # checking if it is a file
            if f.endswith('.jpg'):
                f_name = filename.name.split('.')[0].split('_')[1]
                image = cv2.imread(f)
                db_descriptors[f_name] = histograms.get_block_histograms(image, 7, 40, has_boundingbox, is_query = False, text_mask = None)
        with open(f'{gl
        obal_variables.dir_db_aux}precomputed_descriptors.pkl', 'wb') as handle:
            pickle.dump(db_descriptors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(f'{global_variables.dir_db_aux}precomputed_descriptors.pkl', "rb" ) as f:
                db_descriptors = pickle.load(f)
        except:
            print('Exiting. No precomputed pickle found')
            exit(1)'''

    ## Query processing 
    num_paintings = {}
    mask_coords = {}
    dists = {}
    textbox_coords = {}

    print(f'Start of processing fo the query: {global_variables.dir_query}')
    count=0
    for filename in os.scandir(global_variables.dir_query):
        f = os.path.join(global_variables.dir_query, filename)
        # checking if it is a file
        if f.endswith(f'{global_variables.test_image}.jpg'): 
            f_name = filename.name.split('.')[0]
            image = cv2.imread(f)

            # BG removal
            if(backgrounds):
                # Idea Guillem: query_descriptors[f_name].num_paint, query_descriptors[f_name].mask_coords = mask_v1.generate_masks_otsu(image, f_name, dir_results, may_have_split)
                num_paintings[f_name], mask_coords[f_name] = masks.generate_masks(image, f_name, may_have_split)

            count+=1
            if count==3:
                break

            # ! We are going to change it completely, so it is not necessary to test it
            # ! if(may_have_split):
            # !    split_images.split_images(image, f_name, dir_query)
            """bbox_result = coord_results = []
            text_mask = None
            if(has_boundingbox):
                print('Searching boxes at:', f_name)
                bbox_result, coord_results, text_mask = find_boxes.find_boxes(image, f_name, printbox = True)
                # ! Change this in case of neccessity (inestability of expected text box output)
                textbox_coords[f_name] = bbox_result

            hist_image = histograms.get_block_histograms(image, 7, 40, has_boundingbox, is_query = True, text_mask = text_mask)

            dists[f_name] = distances.query_measures_colour(hist_image, db_descriptors, distance_type)"""
            
    ## Results processing

    """# Results sorting
    results_sorted = utils.get_sorted_list_of_lists_from_dict_of_dicts(dists, distance_type, two_level = may_have_split)
    textboxes_result = utils.get_simple_list_of_lists_from_dict_of_dicts(textbox_coords, two_level = may_have_split)

    # Results printing
    for idx, l in enumerate(results_sorted):
        print(f'For image {idx}:')
        print(f'Search result: {l}')
        if(has_boundingbox): print(f'Boxes: {textboxes_result[idx]}')

    # Results evaluation
    if(solutions):
        # Algorithm evaluation
        mapk1 = scores.mapk(query_solutions, results_sorted, k = 1)
        print(f'The map-1 score is: {round(mapk1, 2)}')
        mapk5 = scores.mapk(query_solutions, results_sorted, k = 5)
        print(f'The map-5 score is: {round(mapk5, 2)}')
        # if(backgrounds):
        #     mask_evaluation.mask_eval_avg(directory_output, dir_query, print_each = False, print_avg = True)
        if(has_boundingbox):
            iou = find_boxes.find_boxes_eval(textboxes_result, boxes_solutions)
            print(f'Mean IoU = {sum(iou)/len(iou)}')

    else:
        print('No solutions given --> Evaluation not avaliable.')

    # Results writing to Pickle file
    with open(f'{global_variables.dir_results}result.pkl', 'wb') as handle:
        pickle.dump(results_sorted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if(has_boundingbox):
        with open(f'{global_variables.dir_results}text_boxes.pkl', 'wb') as handle:
            pickle.dump(textboxes_result, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

if __name__ == "__main__":
    main()
