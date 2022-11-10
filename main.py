"""
    Main file of the project
"""

# External modules
import sys
import os
import cv2
import numpy as np
from numpy.ma.core import mean
import pickle
#from skimage import feature
import statistics
from tqdm import tqdm
from skimage import feature
import re
import time

# Internal modules
import global_variables
import utils
import histograms
import distances
import scores
import masks
import mask_evaluation
import find_boxes
import box_evaluation
import noise
import findText
import distances_analysis
import distances_postprocessing
import rotation

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
        has_backgrounds = bool(utils.str_to_bool(sys.argv[2]))
        has_boundingbox = bool(utils.str_to_bool(sys.argv[3]))
        may_have_split = bool(utils.str_to_bool(sys.argv[4]))
        may_have_noise = bool(utils.str_to_bool(sys.argv[5]))
        may_not_be_in_db = bool(utils.str_to_bool(sys.argv[6]))
        may_have_rotation = bool(utils.str_to_bool(sys.argv[7]))
        solutions = bool(utils.str_to_bool(sys.argv[8]))
        recompute_query = bool(utils.str_to_bool(sys.argv[9]))
        recompute_db = bool(utils.str_to_bool(sys.argv[10]))
    except:
        print(f'Exiting. Not enough arguments ({len(sys.argv) - 1} of 7)')
        exit(1)

    global_variables.init(name_query)
    # Arguments bound checking
    distance_type = global_variables.methods_search['default']['distance_type']

    query_solutions = boxes_solutions = []
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

    start_time = time.time()
    # DB Descriptors and text extraction
    db_descriptors = {}
    if recompute_db:
        print(f'Exctracting descriptors from DB directory: {global_variables.dir_db}')
        for filename in tqdm(os.scandir(global_variables.dir_db)):
            f = os.path.join(global_variables.dir_db, filename)
            # checking if it is a file
            if f.endswith('.jpg'):
                f_name = filename.name.split('.')[0].split('_')[1]
                image = cv2.imread(f)
                db_descriptors[f_name] = histograms.get_block_histograms(image, has_boundingbox, is_query = False, text_mask = None)

        with open(f'{global_variables.dir_db_aux}precomputed_descriptors.pkl', 'wb') as handle:
            pickle.dump(db_descriptors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(f'{global_variables.dir_db_aux}precomputed_descriptors.pkl', "rb" ) as f:
                db_descriptors = pickle.load(f)
        except:
            print('Exiting. No precomputed pickle found')
            exit(1)
    
    time_compute_db = time.time()-start_time


    start_time = time.time()

    ## Query processing 
    num_paintings = {}
    mask_coords = {}
    dists = {}
    textbox_coords = {}
    texts = {}
    coords = []
    loaded_query_dists = False
    rotation_matrix = None

    if not recompute_query:
        # Try to load dists pickle
        try:
            with open(f'{global_variables.dir_results}dists.pkl', "rb" ) as f:
                dists = pickle.load(f)
            # Load also textbox_coords
            with open(f'{global_variables.dir_results}textbox_coords.pkl', "rb" ) as f:
                textbox_coords = pickle.load(f)
            # Load also num_paintings
            with open(f'{global_variables.dir_results}num_paintings.pkl', "rb" ) as f:
                num_paintings = pickle.load(f)
            loaded_query_dists = True
            print('Using precomputed query results')
        except:
            dists = {}
            textbox_coords = {}
            pass
    
    if not loaded_query_dists:
        print(f'Start of processing fo the query: {global_variables.dir_query}')
        for filename in tqdm(os.scandir(global_variables.dir_query)):
            f = os.path.join(global_variables.dir_query, filename)
            # checking if it is a file
            if f.endswith(f'{global_variables.test_image}.jpg'): 
                f_name = filename.name.split('.')[0]
                image = cv2.imread(f)
                im_copy =  image.copy()

                if(may_have_noise):
                    # Image will be a denoised image if needed
                    image = noise.noise_ckeck_removal(image,f_name)

                if(may_have_rotation):
                    # Image will be a rotated image if needed
                    image, rotation_matrix, angle = rotation.rotation_check(image, f_name)

                # In case of no backgrounds (no multiple paintings)
                paintings = [image] 
                f_names = [f_name]

                # BG removal and croping images in paintings
                if(has_backgrounds):
                    f_names = []
                    paintings = []

                    # Idea Guillem: query_descriptors[f_name].num_paint, query_descriptors[f_name].mask_coords = mask_v1.generate_masks_otsu(image, f_name, dir_results, may_have_split)
                    num_paintings[f_name], painting_box = masks.generate_masks_ROT(image, f_name, may_have_split)
                    
                    
                    for paint in range(num_paintings[f_name]):
                        f_names.append(f'{f_name}_part{paint + 1}')
                        mask_coords[f_names[paint]] = painting_box[paint]
                        painting = image[painting_box[paint][1]:painting_box[paint][3], painting_box[paint][0]:painting_box[paint][2]]
                        paintings.append(painting)
                        cv2.imwrite(f'{global_variables.dir_query_aux}{f_names[paint]}.png', painting)
                        c1 = [painting_box[paint][0],painting_box[paint][1]]
                        c2 = [painting_box[paint][2],painting_box[paint][1]]
                        c3 = [painting_box[paint][2],painting_box[paint][3]]
                        c4 = [painting_box[paint][0],painting_box[paint][3]]

                        if(angle != 0):
                            print("rotating coords")
                            c1 = c1 @ rotation_matrix
                            c2 = c2 @ rotation_matrix
                            c3 = c3 @ rotation_matrix
                            c4 = c4 @ rotation_matrix

                        print(c1)
                        print(c2)
                        print(c3)
                        print(c4)

                        mark_red_rectangle = cv2.rectangle(im_copy, (c1[0], c1[1]), (c3[0], c3[1]), (200, 0, 100), 3)
        
                        cv2.namedWindow("lap", cv2.WINDOW_NORMAL)
                        cv2.imshow("lap", im_copy)
                        cv2.waitKey(0)

                                            
                for count, painting in enumerate(paintings):
                    print('\nSearching boxes at:', f_names[count])   
                    if has_boundingbox:
                        coord_results, text_mask, bbox_output = find_boxes.find_boxes_canny(painting, f_names[count])

                        if(has_backgrounds):
                            coords.append([ 
                                            mask_coords[f_names[count]][0] + coord_results[0],
                                            mask_coords[f_names[count]][1] + coord_results[1],
                                            mask_coords[f_names[count]][0] + coord_results[2],
                                            mask_coords[f_names[count]][1] + coord_results[3]
                                        ])
                        else:
                            if (name_query=="qsd1_w2"):
                                coords.append(bbox_output)
                            else:
                                coords.append(coord_results)
                        
                        print('Bounding box coordinates:', coords)
                        # save the text in the dictionary and in the txt file
                        
                        text = findText.getText(coord_results,painting)
                        texts[f_names[count]] = text

                        if(count == 0):
                            with open(f'{global_variables.dir_results}{f_name}.txt', 'w') as f:
                                text = re.sub(r'[^\w\s\n]','',text)
                                f.write(f"{text}")
                        else:
                            with open(f'{global_variables.dir_results}{f_name}.txt', 'a') as f:
                                text = re.sub(r'[^\w\s\n]','',text)
                                f.write(f"{text}")

                        hist_image = histograms.get_block_histograms(painting, has_boundingbox, is_query = True, text_mask = text_mask)
                    else:
                        text = ""
                        hist_image = histograms.get_block_histograms(painting, has_boundingbox, is_query = True, text_mask = None)

                    dists[f_names[count]] = distances.query_measures(hist_image, db_descriptors, distance_type, text)

                if has_boundingbox:
                    textbox_coords[f_name] = coords
                    coords = []
        # Save dists in a pickle
        with open(f'{global_variables.dir_results}dists.pkl', 'wb') as handle:
            pickle.dump(dists, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Save textbox_coords in a pickle
        with open(f'{global_variables.dir_results}textbox_coords.pkl', 'wb') as handle:
            pickle.dump(textbox_coords, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Save num_paintings in a pickle
        with open(f'{global_variables.dir_results}num_paintings.pkl', 'wb') as handle:
            pickle.dump(num_paintings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ## Distances postprocessing
    dists = distances_postprocessing.distances_postprocessing(dists)

    ## Distance analysis
    if may_not_be_in_db:
        distances_analysis.distances_analysis(dists, query_solutions)

    ## Results processing

    # Results sorting, if not two_level num_paintings is {}
    results_sorted = utils.get_sorted_list_of_lists_from_dict_of_dicts(dists, distance_type, num_paintings, two_level = may_have_split, may_not_be_in_db = may_not_be_in_db )
    boxes_predictions = utils.get_simple_list_from_dict(textbox_coords)

    time_compute_query = time.time() - start_time

    print('\n-----RESULTS-----')
    # Results printing
    for idx, l in enumerate(results_sorted):
        print(f'For image {idx}:')
        if(has_boundingbox): print(f'\tBoxes: {boxes_predictions[idx]}')
        print(f'\tSearch result: {l}')
        if(solutions):
            if name_query[-1]!="1":
                apk5 = scores.apk2(query_solutions[idx], results_sorted[idx], k = 5)
            else:
                apk5 = scores.apk(query_solutions[idx], results_sorted[idx], k = 5)
            color = ''
            if apk5 == 1:
                color = global_variables.bcolors.OKGREEN
            elif apk5 > 0:
                color = global_variables.bcolors.WARNING
            else:
                color = global_variables.bcolors.FAIL
            print(f'\tExpected result: {query_solutions[idx]}')
            print(f'\t{color}Apk5 score is: {round(apk5, 2)}{global_variables.bcolors.ENDC}')


    print(f'\n-----EVALUATION of {name_query} using COLOR={global_variables.weights["color"]} / TEXTURE={global_variables.weights["texture"]} / TEXT={global_variables.weights["text"]} / FEATURE={global_variables.weights["feature"]}-----')
    # Results evaluation
    if(solutions):
        if name_query[-1]!='1': # ? for qsdx_w1 ?
            # Algorithm evaluation
            mapk1 = scores.mapk2(query_solutions, results_sorted, k = 1)
            mapk5 = scores.mapk2(query_solutions, results_sorted, k = 5)
        else:
            # Algorithm evaluation
            mapk1 = scores.mapk(query_solutions, results_sorted, k = 1)
            mapk5 = scores.mapk(query_solutions, results_sorted, k = 5)
        print(f'The map-1 score is: {round(mapk1, 3)}')
        print(f'The map-5 score is: {round(mapk5, 3)}')
        # ! Currently not implemented for split paintings
        # if(has_backgrounds):
        #     mask_evaluation.mask_eval_avg(directory_output, dir_query, print_each = False, print_avg = True)
        if(has_boundingbox):
            if(name_query=="qsd1_w2"):
                iou = box_evaluation.find_boxes_eval(boxes_predictions, boxes_solutions)
            else:
                iou = box_evaluation.find_boxes_eval2(boxes_predictions, boxes_solutions)
            print(f'Mean IoU: {round(sum(iou)/len(iou), 2)}')

    else:
        print('No solutions given --> Evaluation not avaliable.')

    # Get time_compute_db and time_compute_query in minutes and seconds
    time_compute_db = time_compute_db/60
    time_compute_query = time_compute_query/60
    time_compute_db_min = int(time_compute_db)
    time_compute_db_sec = int((time_compute_db - time_compute_db_min)*60)
    time_compute_query_min = int(time_compute_query)
    time_compute_query_sec = int((time_compute_query - time_compute_query_min)*60)

    print(f'\n-----EXECUTION TIME of DATABASE: {time_compute_db_min} minutes & {time_compute_db_sec} seconds-----')
    print(f'-----EXECUTION TIME of QUERY: {time_compute_query_min} minutes & {time_compute_query_sec} seconds-----')
    
    # Results writing to Pickle file
    with open(f'{global_variables.dir_results}result.pkl', 'wb') as handle:
        pickle.dump(results_sorted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if(has_boundingbox):
        with open(f'{global_variables.dir_results}text_boxes.pkl', 'wb') as handle:
            pickle.dump(boxes_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
