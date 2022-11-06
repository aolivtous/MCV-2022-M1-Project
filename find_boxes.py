import global_variables
import cv2
import numpy as np

def find_boxes_canny(image, f_name):

    image_cpy = image.copy()
    height, width, _ = image.shape

    # remove noise
    image_blur = cv2.GaussianBlur(image,(11,11),0)

    gray_image = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    gray_image_inv = 255 - gray_image

    #---- apply optimal Canny edge detection using the computed median----
    v = np.median(gray_image)
    sigma = 0.33

    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))

    v_i = np.median(gray_image_inv)

    lower_thresh_i = int(max(0, (1.0 - sigma) * v_i))
    upper_thresh_i = int(min(255, (1.0 + sigma) * v_i))

    # Convolute with proper kernels
    canny = cv2.Canny(gray_image,lower_thresh,upper_thresh,apertureSize=3)
    size_thresh_lapl = 50
    _, canny = cv2.threshold(canny, size_thresh_lapl,255,cv2.THRESH_BINARY)

    canny_i = cv2.Canny(gray_image_inv,lower_thresh_i,upper_thresh_i,apertureSize=3)

    size_thresh_lapl = 50
    _, canny_i = cv2.threshold(canny_i, size_thresh_lapl,255,cv2.THRESH_BINARY)

    # Put the letters together

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    cannyClose = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, element)
    cannyClose_i = cv2.morphologyEx(canny_i, cv2.MORPH_CLOSE, element)


    # Discard the largest connected object, if it is too big (due to the painting drawing lines)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(cannyClose, connectivity=4)
    
    if len(nb_components) > 1:
        max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])

        if(max_size > height*width*0.35):
            cannyClose[output == max_label] = 0    

    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(cannyClose_i, connectivity=4)

    if len(nb_components) > 1:   
        max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])

        if(max_size > height*width*0.35):
            cannyClose_i[output == max_label] = 0    


    # Remove noise

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,11))
    cannyOpen = cv2.morphologyEx(cannyClose, cv2.MORPH_OPEN, element)
    cannyOpen_i = cv2.morphologyEx(cannyClose_i, cv2.MORPH_OPEN, element)

    # Put text together

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image.shape[1]*0.1), 5))
    cannyClose2 = cv2.morphologyEx(cannyOpen, cv2.MORPH_CLOSE, element)
    cannyClose2_i = cv2.morphologyEx(cannyOpen_i, cv2.MORPH_CLOSE, element)

    # Remove noise

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cannyOpen2 = cv2.morphologyEx(cannyClose2, cv2.MORPH_OPEN, element)
    cannyOpen2_i = cv2.morphologyEx(cannyClose2_i, cv2.MORPH_OPEN, element)

    contours_n, _ = cv2.findContours(cannyOpen2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours_i, _ = cv2.findContours(cannyOpen2_i, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_n + contours_i

    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_canny_inicial.png', canny)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_canny_i_inicial.png', canny_i)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_canny_final.png', cannyOpen2)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_canny_i_final.png', cannyOpen2_i)

    image_cpy = image.copy()

    x_min = y_min = w_min = h_min = 0
    best_rectangles = []

    for _, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        mark_brown_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (0, 100, 100), 3)
        if w < int(width * 0.07) or w > int(width * 0.75) or h < int(height * 0.01) or h > int(height*0.5):
            continue
        if h > w*0.55:
            continue
        if h < w*0.05:
            continue

        mark_purple_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (100, 0, 100), 3)

        if (x + (w / 2.0) < (width /2.0) - width * 0.09) or (x + (w / 2.0) > (width / 2.0) + width * 0.09):
            continue

        mark_red_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (0, 0, 255), 3)

        if y + h/2 > height*0.40 and y + h/2 < (height*0.6):
            continue
        mark_blue_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (255, 0, 0), 3)
    
        if h*w > width*height*0.15:
            continue

        mark_white_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (255, 255, 255), 3)

        best_rectangles.append(cnt)
        
    maxScore = 100000000

    for _, cnt in enumerate(best_rectangles):
        x, y, w, h = cv2.boundingRect(cnt)
        dist_centre_x =  abs((x + (w / 2.0)) -  (width /2.0))
        dist_centre_y =  abs((y + (h / 2.0)) -  (height /2.0))
        diag = np.sqrt(w**2+h**2)
        print(dist_centre_x)
        print(dist_centre_y)
        print(diag)

        #lower score the better --> less distance to the centre x , biggest diag and biggest dist from the center y
        #score = dist_centre_x/(diag + dist_centre_y)
        score = dist_centre_x/(diag + dist_centre_y)
        print(score)

        if score < maxScore:
            maxScore = score
            x_min = x
            y_min = y
            w_min = w
            h_min = h


    if x_min != 0 and y_min != 0 and w_min != 0 and h_min != 0:
        x_min = int(x_min - w_min*0.04)
        y_min = int(y_min - h_min*0.15)
        w_min = int(w_min + 2*w_min*0.04)
        h_min = int(h_min + 2*h_min*0.15)

    mark_green_rectangle = cv2.rectangle(image_cpy, (x_min, y_min), (x_min + w_min, y_min + h_min), (0, 255, 0), 3)

    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_final.png', image_cpy)

    text_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(height - 1):
        for j in range(width - 1):
            if j > x_min and i > y_min and j < (x_min + w_min) and i < (y_min + h_min):
                text_mask[i][j] = 0
            else:
                text_mask[i][j] = 255
    
    # Write the text mask
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_text_mask.png', text_mask)

    text_box = [x_min, y_min, x_min + w_min, y_min + h_min]
    bbox_output = [np.array([x_min, y_min]),np.array([x_min, y_min + h_min]),np.array([x_min + w_min, y_min + h_min]),np.array([x_min + w_min, y_min])]

    return text_box, text_mask, bbox_output