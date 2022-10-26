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
#from skimage import feature
import statistics
from tqdm import tqdm
from skimage import feature

# Internal modules
import global_variables
import utils
import histograms
import distances
import scores
import masks
import mask_evaluation
import find_boxes
import noise
import findText

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
        has_backgrounds = bool(utils.str_to_bool(sys.argv[3]))
        has_boundingbox = bool(utils.str_to_bool(sys.argv[4]))
        may_have_split = bool(utils.str_to_bool(sys.argv[5]))
        may_have_noise = bool(utils.str_to_bool(sys.argv[6]))
        solutions = bool(utils.str_to_bool(sys.argv[7]))
        recompute_db = bool(utils.str_to_bool(sys.argv[8]))
    except:
        print(f'Exiting. Not enough arguments ({len(sys.argv) - 1} of 8)')
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

    # DB Descriptors extraction
    db_descriptors = {}
    if recompute_db:
        print(f'Exctracting descriptors from DB directory: {global_variables.dir_db}')
        for filename in tqdm(os.scandir(global_variables.dir_db)):
            f = os.path.join(global_variables.dir_db, filename)
            # checking if it is a file
            if f.endswith('.jpg'):
                f_name = filename.name.split('.')[0].split('_')[1]
                image = cv2.imread(f)
                db_descriptors[f_name] = histograms.get_block_histograms(image, 7, 40, has_boundingbox, is_query = False, text_mask = None, descriptors= ['color','texture'])
        with open(f'{global_variables.dir_db_aux}precomputed_descriptors.pkl', 'wb') as handle:
            pickle.dump(db_descriptors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(f'{global_variables.dir_db_aux}precomputed_descriptors.pkl', "rb" ) as f:
                db_descriptors = pickle.load(f)
        except:
            print('Exiting. No precomputed pickle found')
            exit(1)
    
    ## Query processing 
    num_paintings = {}
    mask_coords = {}
    dists = {}
    textbox_coords = {}
    texts = {}
    to_be_denoised = {}
    coords = []

    print(f'Start of processing fo the query: {global_variables.dir_query}')
    for filename in tqdm(os.scandir(global_variables.dir_query)):
        f = os.path.join(global_variables.dir_query, filename)
        # checking if it is a file
        if f.endswith(f'{global_variables.test_image}.jpg'): 
            f_name = filename.name.split('.')[0]
            image = cv2.imread(f)

            if(may_have_noise):
                to_be_denoised[f_name], image_denoised = noise.noise_ckeck_removal(image,f_name)
                if(to_be_denoised[f_name]):
                    image = image_denoised


            paintings = [image]
            f_names = [f_name]
      
            # BG removal and croping images in paintings
            if(has_backgrounds):
                # Idea Guillem: query_descriptors[f_name].num_paint, query_descriptors[f_name].mask_coords = mask_v1.generate_masks_otsu(image, f_name, dir_results, may_have_split)
                num_paintings[f_name], painting_box = masks.generate_masks(image, f_names[0], may_have_split)
                
                # print(f'num painting image {f_name}: {num_paintings[f_name]}')
                # print(f'top left x ={mask_coords[f_name][0][0]}, top left y ={mask_coords[f_name][0][1]},bottom right x ={mask_coords[f_name][0][2]},bottom right y ={mask_coords[f_name][0][3]}')
              
                if(num_paintings[f_names[0]] == 1):
                    mask_coords[f_names[0]] = painting_box[0]
                    paintings = [image[mask_coords[f_names[0]][1]:mask_coords[f_names[0]][3],mask_coords[f_names[0]][0]:mask_coords[f_names[0]][2]]]
                    cv2.imwrite(f'{global_variables.dir_query_aux}{f_names[0]}.png', paintings[0])   
                   
                elif(num_paintings[f_names[0]] == 2):
                    f_names = [f'{f_names[0]}_part1', f'{f_names[0]}_part2']
                    mask_coords[f_names[0]] = painting_box[0]
                    mask_coords[f_names[1]] = painting_box[1]
                    paintings = [   
                                    image[  mask_coords[f_names[0]][1]:mask_coords[f_names[0]][3],
                                            mask_coords[f_names[0]][0]:mask_coords[f_names[0]][2]
                                        ],
                                    image[  mask_coords[f_names[1]][1]:mask_coords[f_names[1]][3],
                                            mask_coords[f_names[1]][0]:mask_coords[f_names[1]][2]
                                        ]
                                ] 
                    for count, painting in enumerate(paintings):
                        cv2.imwrite(f'{global_variables.dir_query_aux}{f_names[count]}.png', painting)
     
            print('Searching boxes at:', f_name)                                       
            for count, painting in enumerate(paintings):
                print('SubParts ', f_names[count])   
                if has_boundingbox:
                    coord_results, text_mask = find_boxes.find_boxes_lapl(painting, f_names[count], printbox = True)
                   

                    if(has_backgrounds):
                        coords.append([ 
                                        mask_coords[f_names[count]][0] + coord_results[0],
                                        mask_coords[f_names[count]][1] + coord_results[1],
                                        mask_coords[f_names[count]][2] + coord_results[2],
                                        mask_coords[f_names[count]][3] + coord_results[3]
                                    ])
                        
                    # save the text in the dictionary and in the txt file
                    
                    text = findText.getText(coord_results,painting)
                    texts[f_names[count]] = text

                    if(count == 0):
                        with open(f'{global_variables.dir_results}{f_name}.txt', 'w') as f:
                            if(text ==""):
                                f.write('\n')
                            else:
                                f.write(text)
                    else:
                        with open(f'{global_variables.dir_results}{f_name}.txt', 'a') as f:
                            f.write(text)

                    hist_image = histograms.get_block_histograms(painting, 7, 40, has_boundingbox, is_query = True, text_mask = text_mask, descriptors = global_variables.descriptors)
                
     
                else:
                    hist_image = histograms.get_block_histograms(painting, 7, 40, has_boundingbox, is_query = True, text_mask = None, descriptors = global_variables.descriptors)

                dists[f_names[count]] = distances.query_measures_colour(hist_image, db_descriptors, distance_type, descriptors = global_variables.descriptors)

            # ! Change this in case of neccessity (inestability of expected text box output)
            if has_boundingbox and has_backgrounds:
                textbox_coords[f_name] = coords
                coords = []
          
    ## Results processing

    # Results sorting
    results_sorted = utils.get_sorted_list_of_lists_from_dict_of_dicts(dists, distance_type, two_level = may_have_split)
    boxes_predictions = utils.get_simple_list_of_lists_from_dict_of_dicts(textbox_coords, two_level = may_have_split)
 
    # Results printing
    
    for idx, l in enumerate(results_sorted):
        print(f'For image {idx}:')
        print(f'Search result: {l}')
        if(has_boundingbox): print(f'Boxes: {boxes_predictions[idx]}')

    # Results evaluation
    if(solutions):
        # Algorithm evaluation
        mapk1 = scores.mapk(query_solutions, results_sorted, k = 1)
        print(f'The map-1 score is: {round(mapk1, 2)}')
        mapk5 = scores.mapk(query_solutions, results_sorted, k = 5)
        print(f'The map-5 score is: {round(mapk5, 2)}')
        # if(has_backgrounds):
        #     mask_evaluation.mask_eval_avg(directory_output, dir_query, print_each = False, print_avg = True)
        if(has_boundingbox):
            iou = find_boxes.find_boxes_eval2(boxes_predictions, boxes_solutions)
            print(f'Mean IoU: {sum(iou)/len(iou)}')

    else:
        print('No solutions given --> Evaluation not avaliable.')
    
    
    # Results writing to Pickle file
    with open(f'{global_variables.dir_results}result.pkl', 'wb') as handle:
        pickle.dump(results_sorted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if(has_boundingbox):
        with open(f'{global_variables.dir_results}text_boxes.pkl', 'wb') as handle:
            pickle.dump(boxes_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
