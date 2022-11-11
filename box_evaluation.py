import cv2 
import numpy as np
from shapely.geometry import Polygon


#  ? For format of qsd1_w2 text box solutions
def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[0][0], bboxB[0][0])
    yA = max(bboxA[0][1], bboxB[0][1])
    xB = min(bboxA[2][0], bboxB[2][0])
    yB = min(bboxA[2][1], bboxB[2][1])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both bboxes
    bboxAArea = (bboxA[2][0] - bboxA[0][0] + 1) * (bboxA[2][1] - bboxA[0][1] + 1)
    bboxBArea = (bboxB[2][0] - bboxB[0][0] + 1) * (bboxB[2][1] - bboxB[0][1] + 1)
    
    iou = interArea / float(bboxAArea + bboxBArea - interArea)
    
    # return the intersection over union value
    return iou



def find_boxes_eval(list_bbox_prediction, list_bbox_solution):
    iou_list=[]
    # for i in range(len(list_bbox_prediction)):
    #     print('image', i, 'pred', list_bbox_prediction[i])
    for i in range(len(list_bbox_prediction)):
        for j in range(len(list_bbox_prediction[i])):
            try:
                iou = bbox_iou(list_bbox_prediction[i][j], list_bbox_solution[i][j])
                iou_list.append(iou)
                # print(f'iou of image {i} part {j}: {iou}')
            except:
                continue 
    return iou_list



#  ? For format of qsd2_w2 and qsd2_w3 text box solutions
def bbox_iou2(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)
    
    iou = interArea / float(bboxAArea + bboxBArea - interArea)
    
    # return the intersection over union value
    return iou

# For pkl qsd1_w3 that uses this format list-list-tuple [[(1,2,3,4)], [(1,2,3,4),(5,6,7,8)]]
def find_boxes_eval2(list_bbox_prediction, list_bbox_solution):
    iou_list=[]
    # for i in range(len(list_bbox_prediction)):
    #     print('image', i, 'pred', list_bbox_prediction[i])
    for i in range(len(list_bbox_prediction)):
        for j in range(len(list_bbox_prediction[i])):
            try:
                iou = bbox_iou2(list(list_bbox_prediction[i][j]), list(list_bbox_solution[i][j]))
                iou_list.append(iou)
                #print(f'iou of image {i} part {j}: {iou}')
            except:
                continue 
    return iou_list


def frames_eval(list_frames_prediction, list_frames_solution):
    iou_list=[]
    count = 0
    # for i in range(len(list_bbox_prediction)):
    #     print('image', i, 'pred', list_bbox_prediction[i])
    for i in range(len(list_frames_solution)): #loop through images
        for j in range(len(list_frames_solution[i])): #loop through paintings frames
            count +=1
            try:
                # print(f'\n-----------------------Image {i+1} part {j+1} ----------------------------')
                # print(f'Solution={list_frames_solution[i][j][1]}')
                # print(f'Predicti={list_frames_prediction[i][j][1]}')
                rectangle_sol  = Polygon(list_frames_solution[i][j][1])
                rectangle_pred = Polygon(list_frames_prediction[i][j][1])                    

                iou = rectangle_sol.intersection(rectangle_pred).area / rectangle_sol.union(rectangle_pred).area
                iou_list.append(iou)
                # print(f'iou of image {i+1} part {j+1}: {iou}')
            except:
                continue 
    return iou_list,count

