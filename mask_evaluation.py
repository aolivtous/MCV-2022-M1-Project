from main import *
import main

import os

dir_masks_solution='/Users/guillemcapellerafont/Desktop/Master/M1/Projecte/Setmana 1/qsd2_w1'
dir_masks_predicted='/Users/guillemcapellerafont/Desktop/Master/M1/Projecte/Setmana 1/qsd2_w1'

def mask_eval_avg(dir_masks_predicted,dir_masks_solution):
    """
    The function takes in two directories, one with the predicted masks and the other with the solution
    masks. It then iterates through the solution masks and for each mask, it finds the corresponding
    predicted mask and calculates the precision, recall and f1-score. It then averages the precision,
    recall and f1-score for all the masks and returns the average precision, recall and f1-score
    
    :param dir_masks_predicted: The directory where the predicted masks are stored
    :param dir_masks_solution: The directory where the ground truth masks are stored
    :return: the average precision, recall and f1-score for all the masks in the directory.
    """
    avg_precision=0
    avg_recall=0
    avg_f1=0 
    count=0
    
    for filename in os.scandir(dir_masks_solution):
        f_solution = os.path.join(dir_masks_solution, filename)
        # checking if it is a file
        if f_solution.endswith('.png'):
            # Splitting the file name and getting the file name without the extension.
            split_f = f_solution.split('/')[-1]
            f_name_solution = split_f.split('.')[0]
            f_name_solution = f_name_solution[-5:]

            f_predicted = dir_masks_predicted + '/'+ split_f
            print("Mask #{}  Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}",f_name_solution, precision, recall, f1)

            precision, recall, f1 = mask_eval(f_predicted,f_solution)
            count=count+1

            avg_precision +=precision
            avg_recall +=recall
            avg_f1 +=f1
    
    avg_precision = avg_precision / count
    avg_recall = avg_recall / count
    avg_f1 = avg_f1 / count
    
    return avg_precision, avg_recall, avg_f1



def mask_eval(path_mask_predicted,path_mask_solution):
    """
    It takes two images, one of which is the predicted mask and the other is the ground truth mask, and
    returns the precision, recall and f1 score
    
    :param path_mask_predicted: path to the predicted mask
    :param path_mask_solution: the path to the ground truth mask
    :return: precision, recall, f1
    """
    mask_predicted = cv2.imread(path_mask_predicted,0) /255
    mask_predicted = mask_predicted.reshape(-1)
    
    mask_solution = cv2.imread(path_mask_solution,0) /255
    mask_solution = mask_solution.reshape(-1)
    
    tp = np.dot(mask_solution, mask_predicted) 
    tn = np.dot(1-mask_solution, 1-mask_predicted) 
    fp = np.dot(1-mask_solution, mask_predicted) 
    fn = np.dot(mask_solution, 1-mask_predicted) 

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(precision*recall)/(precision+recall)

    return precision,recall,f1


