import global_variables
import cv2
import numpy as np

def color_factor(image_rgb):
    height,width,channels = image_rgb.shape
    diff = 0
    for i in range(height):
        for j in range(width):
            rg = abs(int(image_rgb[i,j][0])-int(image_rgb[i,j][1]))
            rb = abs(int(image_rgb[i,j][0])-int(image_rgb[i,j][2]))
            gb = abs(int(image_rgb[i,j][1])-int(image_rgb[i,j][2]))
            diff += rg+rb+gb
    
    return diff / (width*height)


def find_boxes(image, f_name, printbox=False):
    bbox_output = []
    aux_bbox_output = []
    result = []
    aux_result = []
    is_part = False
    
    if 'part' in f_name:
        is_part = True
    else:
        is_part = False

    height, width, channels = 0, 0, 0
    try:
        height, width, channels = image.shape
    except:
        print(f_name)
        cv2.imshow('test', image)
        cv2.waitKey(0)

    #image_blur = cv2.GaussianBlur(image,(3,3),0)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bin_image = np.zeros((height, width), dtype=np.uint8)

    image_th = np.zeros((height, width))
    for i in range(height - 1):
        for j in range(width - 1):
            if image_hsv[i][j][1] * (1/2.55) > 8 or (image_hsv[i][j][2] * (1/2.55) > 40 and image_hsv[i][j][2] * (1/2.55) < 60): # ! Provar sense or
                bin_image[i][j] = 0
            else:
                bin_image[i][j] = 255
    

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.05), 2))
    bin_image_close = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_image_close = cv2.morphologyEx(bin_image_close, cv2.MORPH_OPEN, element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.25), 2))
    bin_image_close_close = cv2.morphologyEx(bin_image_close, cv2.MORPH_OPEN, element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_image_close_close = cv2.morphologyEx(bin_image_close_close, cv2.MORPH_OPEN, element)

    retr_mode = cv2.RETR_EXTERNAL
    contours_close, hierarchy = cv2.findContours(bin_image_close, retr_mode,cv2.CHAIN_APPROX_SIMPLE)
    contours_close_close, hierarchy = cv2.findContours(bin_image_close_close, retr_mode,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(bin_image, retr_mode,cv2.CHAIN_APPROX_SIMPLE)


    contours = contours + contours_close + contours_close_close

    image_cpy = image.copy()
    mindiff = 1000000000
    x_min = y_min = w_min = h_min = 0
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w < int(width * 0.05) or h < int(height * 0.01) or h > int(height*0.5):
            continue
        if h*w < (height*width)*0.0005:
            continue
        if h > w:
            continue
        if (x + (w / 2.0) < (width /2.0) - width * 0.03) or (x + (w / 2.0) > (width / 2.0) + width * 0.03):
            continue

        mark_red_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (0, 0, 255), 3)

        diff = color_factor(image_rgb[y:y + h, x:x + w]) # [y:y + h, x:x + w]
        if diff < mindiff:
            mindiff=diff
            x_min = x
            y_min = y
            w_min = w
            h_min = h
   
    #Extension of rectangle 
    tol = 10
    try:
        while abs(int(image_hsv[y_min + h_min, x_min][1]) - int(image_hsv[y_min + h_min +1, x_min][1])) < tol and (image_hsv[y_min + h_min, x_min][2]<25 or image_hsv[y_min + h_min, x_min][2]>240) :
            h_min = h_min + 1
    except:
        pass
    
    try:
        while abs(int(image_hsv[y_min + h_min, x_min][1]) - int(image_hsv[y_min + h_min, x_min - 1][1])) < tol and (image_hsv[y_min + h_min, x_min][2]<25 or image_hsv[y_min + h_min, x_min][2]>240) :
            x_min = x_min - 1
    except:
        pass
    
    try:
        while abs(int(image_hsv[y_min, x_min + w_min][1]) - int(image_hsv[y_min, x_min + w_min + 1][1])) < tol and (image_hsv[y_min, x_min + w_min][2]<25 or image_hsv[y_min, x_min + w_min][2]>240):
            w_min = w_min + 1
    except:
        pass
    
    try:
        while abs(int(image_hsv[y_min, x_min + w_min][1]) - int(image_hsv[y_min - 1 , x_min + w_min][1])) < tol and (image_hsv[y_min, x_min + w_min][2]<25 or image_hsv[y_min, x_min + w_min][2]>240):
            y_min = y_min - 1
    except:
        pass

        
    #print(mindiff)
    mark_green_rectangle = cv2.rectangle(image_cpy, (x_min, y_min), (x_min + w_min, y_min + h_min), (0, 255, 0), 3)
    
    if printbox:
        cv2.imwrite(f'{global_variables.dir_query_aux}{f_name}_bin.png', bin_image)
        cv2.imwrite(f'{global_variables.dir_query_aux}{f_name}_bin_close.png', bin_image_close)
        cv2.imwrite(f'{global_variables.dir_query_aux}{f_name}_bin_close_close.png', bin_image_close_close)
        cv2.imwrite(f'{global_variables.dir_query_aux}{f_name}_rectangle_exten.png', image_cpy)
    
    text_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(height - 1):
        for j in range(width - 1):
            if j > x_min and i > y_min and j < (x_min + w_min) and i < (y_min + h_min):
                text_mask[i][j] = 0
            else:
                text_mask[i][j] = 255
    #th, box_mask_bi = cv2.threshold(box_mask, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'{global_variables.dir_query_aux}{f_name}_bin_box.png', text_mask)

    # if(is_part):
    #     aux_result.append([x_min, y_min, x_min+w_min, y_min+h_min])
    #     aux_bbox_output.append([np.array([x_min, y_min]),np.array([x_min, y_min + h_min]),np.array([x_min + w_min, y_min + h_min]),np.array([x_min + w_min, y_min])])
    #     if f_name.endswith('2'):
    #         result.append(aux_result)   
    #         bbox_output.append(aux_bbox_output)
    #         aux_result = []
    #         aux_bbox_output = []
    # else:

    #result = [(x_min, y_min, x_min+w_min, y_min+h_min)]   
    #bbox_output = [[np.array([x_min, y_min]),np.array([x_min, y_min + h_min]),np.array([x_min + w_min, y_min + h_min]),np.array([x_min + w_min, y_min])]]
    result = [x_min, y_min, x_min+w_min, y_min+h_min]      

    #return bbox_output, result, 
    #return bbox_output, result
    return result, text_mask


def find_boxes_lapl(image, f_name, printbox=False):

    image_cpy = image.copy()
    width, height, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # remove noise
    image_blur = cv2.GaussianBlur(image,(11,11),0)

    
    gray_image = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    gray_image_inv = 255 - gray_image

    # convolute with proper kernels
    lap = cv2.Laplacian(gray_image,cv2.CV_32F, ksize = 3)
    lap_inv = cv2.Laplacian(gray_image_inv,cv2.CV_32F, ksize = 3)

    
    size_thresh_lapl = 8
    th, laplacian = cv2.threshold(lap, size_thresh_lapl,255,cv2.THRESH_BINARY)
    th, laplacian_inv = cv2.threshold(lap_inv, size_thresh_lapl,255,cv2.THRESH_BINARY)

    # cv2.imshow("Laplacian",laplacian)
    # cv2.waitKey(0)

    # cv2.imshow("Laplacian_inv",laplacian_inv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width, channels = image.shape
    pixels_saturated = np.zeros((height, width), dtype=np.uint8)
    
    size_thresh_s = 30
    for i in range(height - 1):
        for j in range(width - 1):
            if hsv_image[i][j][1] > size_thresh_s: # ! Provar sense or
                pixels_saturated[i][j] = 0
            else:
                pixels_saturated[i][j] = 255

    # cv2.imshow("saturated5",pixels_saturated)
    # cv2.waitKey(0) 
    
    
    pixels_saturated=abs((hsv_image[:,:,1]>size_thresh_s)-1)
    
    laplacian_wtht_sat = (laplacian * pixels_saturated).astype(np.uint8)
    laplacian_inv_wtht_sat = (laplacian_inv * pixels_saturated).astype(np.uint8)

 
    # cv2.imshow("Laplacian_wtht_sat", laplacian_wtht_sat)
    # cv2.waitKey(0)

    # cv2.imshow("Laplacian_inv_wtht_sat",laplacian_inv_wtht_sat)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # Apply close morphology operator


    #element = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))


    #laplacian_close = cv2.morphologyEx(laplacian, cv2.MORPH_CLOSE, element)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.1), 3))
    laplacian_close = cv2.morphologyEx(laplacian_wtht_sat,cv2.MORPH_CLOSE,element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    laplacian_open = cv2.morphologyEx(laplacian_close, cv2.MORPH_OPEN, element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.1), 3))
    laplacian_close_inv = cv2.morphologyEx(laplacian_inv_wtht_sat,cv2.MORPH_CLOSE,element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    laplacian_open_inv = cv2.morphologyEx(laplacian_close_inv, cv2.MORPH_OPEN, element)


    # cv2.imshow("Laplacian closing", laplacian_close)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("Laplacian opening", laplacian_open)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("Laplacian closing", laplacian_close_inv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("Laplacian opening", laplacian_open_inv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    mask_open2 = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, element2)

    cv2.imshow("Laplacian", mask_open2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    contours1, hierarchy = cv2.findContours(laplacian_open, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy = cv2.findContours(laplacian_open_inv, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    contours = contours1 + contours2
    #print(len(contours))

    """
    areaArray = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    #first sort the array by area
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    #find the nth largest contours

    firstgestcontour = sorteddata[0][1]
    x, y, w, h = cv2.boundingRect(firstgestcontour)
    mark_red_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (0, 0, 255), 3)
    text_box = [x, y, x+w, y+h]"""

    image_cpy = image.copy()
    mindiff = 1000000000
    x_min = y_min = w_min = h_min = 0
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w < int(width * 0.05) or h < int(height * 0.01) or h > int(height*0.5):
            continue
        if h*w < (height*width)*0.0005:
            continue
        if h > w:
            continue
        if (x + (w / 2.0) < (width /2.0) - width * 0.03) or (x + (w / 2.0) > (width / 2.0) + width * 0.03):
            continue

        mark_red_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (0, 0, 255), 3)

        diff = color_factor(rgb_image[y:y + h, x:x + w]) # [y:y + h, x:x + w]
        if diff < mindiff:
            mindiff=diff
            x_min = x
            y_min = y
            w_min = w
            h_min = h

    mark_green_rectangle = cv2.rectangle(image_cpy, (x_min, y_min), (x_min + w_min, y_min + h_min), (0, 255, 0), 3)
    # cv2.imshow("Final", image_cpy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    text_box = [x_min, y_min, x_min+w_min, y_min+h_min]
    print (text_box)

    """cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_text_laplacian.png', laplacian)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_text_laplacian_open.png', np.uint8(mask_open))
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_text_laplaian_open_dilate.png', np.uint8(dilation))
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_text_laplaian_open_dilate_open.png', np.uint8(mask_open2))"""
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_text_laplacian_boxes.png', image_cpy)

    text_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(height - 1):
        for j in range(width - 1):
            if j > x and i > y and j < (x + w) and i < (y + h):
                text_mask[i][j] = 0
            else:
                text_mask[i][j] = 255

    return text_box, text_mask


def find_boxes2(image, f_name, printbox=False):
    bbox_output = []
    aux_bbox_output = []
    result = []
    aux_result = []
    is_part = False
    
    if 'part' in f_name:
        is_part = True
    else:
        is_part = False

    height, width, channels = 0, 0, 0
    try:
        height, width, channels = image.shape
    except:
        print(f_name)
        cv2.imshow('test', image)
        cv2.waitKey(0)

    cv2.imshow('image', image)
    cv2.waitKey(0)

    image_blur = cv2.medianBlur(image,3)
    gray_image = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray_image,cv2.CV_32F, ksize = 3)
    th, laplacian = cv2.threshold(lap, 5,255,cv2.THRESH_BINARY)

    cv2.imshow('laplacian', laplacian)
    cv2.waitKey(0)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bin_image = np.zeros((height, width), dtype=np.uint8)

    image_th = np.zeros((height, width))
    for i in range(height - 1):
        for j in range(width - 1):
            if image_hsv[i][j][1] * (1/2.55) > 8 or (image_hsv[i][j][2] * (1/2.55) > 40 and image_hsv[i][j][2] * (1/2.55) < 60): # ! Provar sense or
                bin_image[i][j] = 0
            else:
                bin_image[i][j] = 255
    

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.1),2))
    top_hat_x = cv2.morphologyEx(bin_image, cv2.MORPH_TOPHAT, element)

    cv2.imshow('top_hat_x', top_hat_x)
    cv2.waitKey(0)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    top_hat_x = cv2.morphologyEx(top_hat_x, cv2.MORPH_OPEN, element)

    cv2.imshow('top_hat_x_opening', top_hat_x)
    cv2.waitKey(0)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.1),2))
    black_hat_x = cv2.morphologyEx(bin_image, cv2.MORPH_BLACKHAT, element)
    
    cv2.imshow('black_hat_x', black_hat_x)
    cv2.waitKey(0)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    black_hat_x = cv2.morphologyEx(black_hat_x, cv2.MORPH_CLOSE, element)

    cv2.imshow('black_hat_x_close', black_hat_x)
    cv2.waitKey(0)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.05), 2))
    bin_image_close = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_image_close = cv2.morphologyEx(bin_image_close, cv2.MORPH_OPEN, element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.25), 2))
    bin_image_close_close = cv2.morphologyEx(bin_image_close, cv2.MORPH_OPEN, element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_image_close_close = cv2.morphologyEx(bin_image_close_close, cv2.MORPH_OPEN, element)

    retr_mode = cv2.RETR_EXTERNAL
    contours_close, hierarchy = cv2.findContours(bin_image_close, retr_mode,cv2.CHAIN_APPROX_SIMPLE)
    contours_close_close, hierarchy = cv2.findContours(bin_image_close_close, retr_mode,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(bin_image, retr_mode,cv2.CHAIN_APPROX_SIMPLE)
    contours_top_hat, hierarchy = cv2.findContours(top_hat_x, retr_mode,cv2.CHAIN_APPROX_SIMPLE)
    contours_black_hat, hierarchy = cv2.findContours(black_hat_x, retr_mode,cv2.CHAIN_APPROX_SIMPLE)
    

    contours = contours + contours_close + contours_close_close + contours_top_hat + contours_black_hat

    image_cpy = image.copy()
    mindiff = 1000000000
    x_min = y_min = w_min = h_min = 0
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w < int(width * 0.05) or h < int(height * 0.01) or h > int(height*0.5):
            continue
        if h*w < (height*width)*0.0005:
            continue
        if h > w:
            continue
        if (x + (w / 2.0) < (width /2.0) - width * 0.03) or (x + (w / 2.0) > (width / 2.0) + width * 0.03):
            continue

        mark_red_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (0, 0, 255), 3)

        diff = color_factor(image_rgb[y:y + h, x:x + w]) # [y:y + h, x:x + w]
        if diff < mindiff:
            mindiff=diff
            x_min = x
            y_min = y
            w_min = w
            h_min = h
   
    #Extension of rectangle 
    tol = 10
    try:
        while abs(int(image_hsv[y_min + h_min, x_min][1]) - int(image_hsv[y_min + h_min +1, x_min][1])) < tol and (image_hsv[y_min + h_min, x_min][2]<25 or image_hsv[y_min + h_min, x_min][2]>240) :
            h_min = h_min + 1
    except:
        pass
    
    try:
        while abs(int(image_hsv[y_min + h_min, x_min][1]) - int(image_hsv[y_min + h_min, x_min - 1][1])) < tol and (image_hsv[y_min + h_min, x_min][2]<25 or image_hsv[y_min + h_min, x_min][2]>240) :
            x_min = x_min - 1
    except:
        pass
    
    try:
        while abs(int(image_hsv[y_min, x_min + w_min][1]) - int(image_hsv[y_min, x_min + w_min + 1][1])) < tol and (image_hsv[y_min, x_min + w_min][2]<25 or image_hsv[y_min, x_min + w_min][2]>240):
            w_min = w_min + 1
    except:
        pass
    
    try:
        while abs(int(image_hsv[y_min, x_min + w_min][1]) - int(image_hsv[y_min - 1 , x_min + w_min][1])) < tol and (image_hsv[y_min, x_min + w_min][2]<25 or image_hsv[y_min, x_min + w_min][2]>240):
            y_min = y_min - 1
    except:
        pass

        
    #print(mindiff)
    mark_green_rectangle = cv2.rectangle(image_cpy, (x_min, y_min), (x_min + w_min, y_min + h_min), (0, 255, 0), 3)
    
    if printbox:
        cv2.imwrite(f'{global_variables.dir_query_aux}{f_name}_bin.png', bin_image)
        cv2.imwrite(f'{global_variables.dir_query_aux}{f_name}_bin_close.png', bin_image_close)
        cv2.imwrite(f'{global_variables.dir_query_aux}{f_name}_bin_close_close.png', bin_image_close_close)
        cv2.imwrite(f'{global_variables.dir_query_aux}{f_name}_rectangle_exten.png', image_cpy)
    
    text_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(height - 1):
        for j in range(width - 1):
            if j > x_min and i > y_min and j < (x_min + w_min) and i < (y_min + h_min):
                text_mask[i][j] = 0
            else:
                text_mask[i][j] = 255
    #th, box_mask_bi = cv2.threshold(box_mask, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'{global_variables.dir_query_aux}{f_name}_bin_box.png', text_mask)

    # if(is_part):
    #     aux_result.append([x_min, y_min, x_min+w_min, y_min+h_min])
    #     aux_bbox_output.append([np.array([x_min, y_min]),np.array([x_min, y_min + h_min]),np.array([x_min + w_min, y_min + h_min]),np.array([x_min + w_min, y_min])])
    #     if f_name.endswith('2'):
    #         result.append(aux_result)   
    #         bbox_output.append(aux_bbox_output)
    #         aux_result = []
    #         aux_bbox_output = []
    # else:
    result = [(x_min, y_min, x_min+w_min, y_min+h_min)]   
    bbox_output = [[np.array([x_min, y_min]),np.array([x_min, y_min + h_min]),np.array([x_min + w_min, y_min + h_min]),np.array([x_min + w_min, y_min])]]
          
    return bbox_output, result, text_mask








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


# boxes_predicted = [[(0, 0, 0, 0)], [(110, 40, 605, 82)], [(424, 1232, 472, 1248)], [(0, 0, 0, 0)], [(230, 73, 259, 81)], [(0, 0, 0, 0)], [(119, 14, 706, 116)], [(175, 486, 415, 494)], [(35, 136, 204, 156)], [(57, 265, 319, 301)], [(54, 10, 377, 60)], [(674, 1712, 1269, 1776)], [(39, 10, 217, 52)], [(77, 19, 429, 59)], [(0, 525, 724, 703)], [(75, 328, 410, 334)], [(22, 8, 414, 134)], [(0, 0, 0, 0)], [(316, 177, 456, 212)], [(61, 48, 310, 78)], [(54, 325, 265, 348)], [(0, 357, 1436, 481)], [(78, 320, 430, 368)], [(68, 82, 1531, 512)], [(0, 0, 0, 0)], [(0, 0, 0, 0)], [(82, 560, 457, 607)], [(64, 18, 357, 48)], [(0, 0, 0, 0)], [(144, 438, 217, 471)]]
# boxes_solutions = [[(51, 39, 292, 73)], [(45, 21, 257, 65)], [(418, 79, 1018, 214)], [(76, 317, 432, 372)], [(38, 10, 216, 52)], [(72, 325, 412, 385)], [(114, 490, 646, 565)], [(64, 360, 365, 404)], [(35, 136, 202, 153)], [(672, 1703, 1272, 1787)], [(75, 17, 429, 60)], [(108, 576, 615, 668)], [(55, 41, 314, 85)], [(65, 8, 369, 60)], [(62, 4, 355, 65)], [(80, 558, 457, 609)], [(47, 324, 265, 369)], [(512, 220, 1112, 300)], [(170, 80, 770, 200)], [(107, 38, 605, 90)], [(56, 265, 318, 301)], [(66, 14, 376, 62)], [(90, 487, 509, 571)], [(120, 17, 679, 119)], [(72, 520, 411, 581)], [(86, 41, 490, 81)], [(74, 67, 419, 119)], [(57, 417, 323, 449)], [(63, 432, 357, 474)], [(65, 8, 369, 69)]]



