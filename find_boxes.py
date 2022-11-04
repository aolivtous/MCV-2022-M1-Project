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
    

    """cv2.imshow("Laplacian",laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    cv2.imshow("Laplacian_inv",laplacian_inv)
    cv2.waitKey(0)
    cv2.destroyAllWindows"""


    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # laplacian = cv2.morphologyEx(laplacian, cv2.MORPH_CLOSE, element)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    laplacian = cv2.morphologyEx(laplacian, cv2.MORPH_OPEN, element)

    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # laplacian_inv = cv2.morphologyEx(laplacian_inv, cv2.MORPH_CLOSE, element)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    laplacian_inv = cv2.morphologyEx(laplacian_inv, cv2.MORPH_OPEN, element)

    # cv2.imshow("Laplacian",laplacian)
    # cv2.waitKey(0)

    # cv2.imshow("Laplacian_inv",laplacian_inv)
    # cv2.waitKey(0)


    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width, channels = image.shape
    pixels_saturated = np.zeros((height, width), dtype=np.uint8)
    
    size_thresh_s = 30
    # for i in range(height - 1):
    #     for j in range(width - 1):
    #         if hsv_image[i][j][1] > size_thresh_s: # ! Provar sense or
    #             pixels_saturated[i][j] = 0
    #         else:
    #             pixels_saturated[i][j] = 255

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
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.10), 3))
    laplacian_close = cv2.morphologyEx(laplacian_wtht_sat,cv2.MORPH_CLOSE,element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    laplacian_open = cv2.morphologyEx(laplacian_close, cv2.MORPH_OPEN, element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.10), 3))
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
    x_min1 = y_min1 = w_min1 = h_min1 = 0
    x_min2 = y_min2 = w_min2 = h_min2 = 0

    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w < int(width * 0.05) or h < int(height * 0.01) or h > int(height*0.5):
            continue
        if h*w < (height*width)*0.005:
            continue
        if h > w*0.75:
            continue
        if h < w*0.05:
            continue
        if (x + (w / 2.0) < (width /2.0) - width * 0.03) or (x + (w / 2.0) > (width / 2.0) + width * 0.03):
            continue
        # if (y > height * 0.2):
        #     if (y + h > height*0.85):
        #         continue
        # else:
        #     continue

        mark_white_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (255, 255, 255), 3)

        diff = color_factor(rgb_image[y:y + h, x:x + w]) # [y:y + h, x:x + w]
        if diff < mindiff:
            mindiff=diff
            x_min1 = x
            y_min1 = y
            w_min1 = w
            h_min1 = h

    mindiff = 1000000000
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if (x==x_min1 and y==y_min1) and (w==w_min1 and h==h_min1):
            continue
        if w < int(width * 0.05) or h < int(height * 0.01) or h > int(height*0.5):
            continue
        if h*w < (height*width)*0.005:
            continue
        if h > w*0.75:
            continue
        if h < w*0.05:
            continue
        if (x + (w / 2.0) < (width /2.0) - width * 0.03) or (x + (w / 2.0) > (width / 2.0) + width * 0.03):
            continue
        # if (y > height * 0.2):
        #     if (y + h > height*0.85):
        #         continue
        # else:
        #     continue

        diff = color_factor(rgb_image[y:y + h, x:x + w]) # [y:y + h, x:x + w]
        if diff < mindiff:
            mindiff=diff
            x_min2 = x
            y_min2 = y
            w_min2 = w
            h_min2 = h

    x_min = x_min1
    y_min = y_min1
    w_min = w_min1
    h_min = h_min1
    area1 = w_min1 * h_min1
    area2 = w_min2 * h_min2
    if area2 > area1 and abs(y_min1 - y_min2) < 20 and abs(x_min1 - x_min2) < 20:
        x_min = x_min2
        y_min = y_min2
        w_min = w_min2
        h_min = h_min2



    # Extension & reduction of the rectangle
    # tol = 2
    # try:
    #     while abs(gray_image[y_min + h_min, x_min + int(w_min/2)] - gray_image[y_min + h_min +1, x_min + int(w_min/2)]) < tol :
    #         h_min = h_min + 1
    #         print('Extension down')
    # except:
    #     pass
    # tol = 2
    # try:
    #     while abs(gray_image[y_min + h_min, x_min + int(w_min/2)] - gray_image[y_min - 1, x_min + int(w_min/2)]) < tol :
    #         y_min = y_min -1
    #         print('Extension up')
    # except:
    #     pass
    # tol = 2
    # try:
    #     while abs(gray_image[y_min+ int(h_min/2), x_min] - gray_image[y_min+ int(h_min/2), x_min -1 ]) < tol :
    #         x_min = x_min - 1
    #         print('Extension left')
    # except:
    #     pass
    # tol = 2
    # try:
    #     while abs(gray_image[y_min+ int(h_min/2), x_min+w_min] - gray_image[y_min+ int(h_min/2), x_min+w_min+1 ]) < tol :
    #         w_min = w_min +1
    #         print('Extension right')
    # except:
    #     pass

    mark_red_rectangle = cv2.rectangle(image_cpy, (x_min1, y_min1), (x_min1 + w_min1, y_min1 + h_min1), (0, 0, 255), 3)
    mark_blue_rectangle = cv2.rectangle(image_cpy, (x_min2, y_min2), (x_min2 + w_min2, y_min2 + h_min2), (255, 0, 0), 3)
    mark_green_rectangle = cv2.rectangle(image_cpy, (x_min, y_min), (x_min + w_min, y_min + h_min), (0, 255, 0), 3)
    # cv2.imshow("Final", image_cpy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    text_box = [x_min, y_min, x_min+w_min, y_min+h_min]
    #print (text_box)

    
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_blur1.png', image_blur)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_laplace_bin_open2.png', laplacian)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_laplace_inv_bin_open2.png', laplacian_inv)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_laplace_wtht_saturated3.png', laplacian_wtht_sat)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_laplace_inv_wtht_saturated3.png', laplacian_inv_wtht_sat)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_laplace_close-open4.png', laplacian_open)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_laplace_inv_close-open4.png', laplacian_open_inv)
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

    bbox_output = [np.array([x_min, y_min]),np.array([x_min, y_min + h_min]),np.array([x_min + w_min, y_min + h_min]),np.array([x_min + w_min, y_min])]

    return text_box, text_mask, bbox_output




def find_boxes_canny(image, f_name, printbox=False):

    image_cpy = image.copy()
    height, width, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # remove noise
    image_blur = cv2.GaussianBlur(image,(11,11),0)

    cv2.namedWindow("blur", cv2.WINDOW_NORMAL) 
    cv2.imshow("blur",image_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    
    gray_image = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    gray_image_inv = 255 - gray_image

    cv2.namedWindow("inv", cv2.WINDOW_NORMAL) 
    cv2.imshow("inv",gray_image_inv)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    # convolute with proper kernels
    canny = cv2.Canny(gray_image_inv,100,200)
    size_thresh_lapl = 20
    th, canny = cv2.threshold(canny, size_thresh_lapl,255,cv2.THRESH_BINARY)

    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows


    
    size_thresh_lapl = 10
    th, canny = cv2.threshold(canny, size_thresh_lapl,255,cv2.THRESH_BINARY)

    '''cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows'''



    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, element)
    #element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #laplacian = cv2.morphologyEx(laplacian, cv2.MORPH_OPEN, element)

    '''cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",canny)
    cv2.waitKey(0)'''

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width, channels = image.shape
    pixels_saturated = np.zeros((height, width), dtype=np.uint8)
    
    '''size_thresh_s = 30
    # for i in range(height - 1):
    #     for j in range(width - 1):
    #         if hsv_image[i][j][1] > size_thresh_s: # ! Provar sense or
    #             pixels_saturated[i][j] = 0
    #         else:
    #             pixels_saturated[i][j] = 255

    # cv2.imshow("saturated5",pixels_saturated)
    # cv2.waitKey(0) 
    
    
    pixels_saturated=abs((hsv_image[:,:,1]>size_thresh_s)-1)
    
    canny_wtht_sat = (canny * pixels_saturated).astype(np.uint8)
    
 
    cv2.imshow("Canny", canny_wtht_sat)

    cv2.waitKey(0)'''


    # Apply close morphology operator


    #element = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))


    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(height*0.05),3))
    canny_close = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, int(height*0.05)))
    canny_close = cv2.morphologyEx(canny_close, cv2.MORPH_CLOSE, element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.20), 5))
    canny_open = cv2.morphologyEx(canny_close, cv2.MORPH_OPEN, element)



    '''cv2.imshow("Canny", canny_close)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imshow("Canny", canny_open)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''


    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #print(len(contours))


    image_cpy = image.copy()
    mindiff = 1000000000
    x_min1 = y_min1 = w_min1 = h_min1 = 0
    x_min2 = y_min2 = w_min2 = h_min2 = 0

    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w < int(width * 0.05) or h < int(height * 0.01) or h > int(height*0.5):
            continue
        if h*w < (height*width)*0.005:
            continue
        if h > w*0.75:
            continue
        if h < w*0.05:
            continue
        if (x + (w / 2.0) < (width /2.0) - width * 0.03) or (x + (w / 2.0) > (width / 2.0) + width * 0.03):
            continue
        # if (y > height * 0.2):
        #     if (y + h > height*0.85):
        #         continue
        # else:
        #     continue

        mark_white_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (255, 255, 255), 3)

        diff = color_factor(rgb_image[y:y + h, x:x + w]) # [y:y + h, x:x + w]
        if diff < mindiff:
            mindiff=diff
            x_min1 = x
            y_min1 = y
            w_min1 = w
            h_min1 = h

    mindiff = 1000000000
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if (x==x_min1 and y==y_min1) and (w==w_min1 and h==h_min1):
            continue
        if w < int(width * 0.05) or h < int(height * 0.01) or h > int(height*0.5):
            continue
        if h*w < (height*width)*0.005:
            continue
        if h > w*0.75:
            continue
        if h < w*0.05:
            continue
        if (x + (w / 2.0) < (width /2.0) - width * 0.03) or (x + (w / 2.0) > (width / 2.0) + width * 0.03):
            continue
        # if (y > height * 0.2):
        #     if (y + h > height*0.85):
        #         continue
        # else:
        #     continue

        diff = color_factor(rgb_image[y:y + h, x:x + w]) # [y:y + h, x:x + w]
        if diff < mindiff:
            mindiff=diff
            x_min2 = x
            y_min2 = y
            w_min2 = w
            h_min2 = h

    x_min = x_min1
    y_min = y_min1
    w_min = w_min1
    h_min = h_min1
    area1 = w_min1 * h_min1
    area2 = w_min2 * h_min2
    if area2 > area1 and abs(y_min1 - y_min2) < 20 and abs(x_min1 - x_min2) < 20:
        x_min = x_min2
        y_min = y_min2
        w_min = w_min2
        h_min = h_min2



    # Extension & reduction of the rectangle
    # tol = 2
    # try:
    #     while abs(gray_image[y_min + h_min, x_min + int(w_min/2)] - gray_image[y_min + h_min +1, x_min + int(w_min/2)]) < tol :
    #         h_min = h_min + 1
    #         print('Extension down')
    # except:
    #     pass
    # tol = 2
    # try:
    #     while abs(gray_image[y_min + h_min, x_min + int(w_min/2)] - gray_image[y_min - 1, x_min + int(w_min/2)]) < tol :
    #         y_min = y_min -1
    #         print('Extension up')
    # except:
    #     pass
    # tol = 2
    # try:
    #     while abs(gray_image[y_min+ int(h_min/2), x_min] - gray_image[y_min+ int(h_min/2), x_min -1 ]) < tol :
    #         x_min = x_min - 1
    #         print('Extension left')
    # except:
    #     pass
    # tol = 2
    # try:
    #     while abs(gray_image[y_min+ int(h_min/2), x_min+w_min] - gray_image[y_min+ int(h_min/2), x_min+w_min+1 ]) < tol :
    #         w_min = w_min +1
    #         print('Extension right')
    # except:
    #     pass

    mark_red_rectangle = cv2.rectangle(image_cpy, (x_min1, y_min1), (x_min1 + w_min1, y_min1 + h_min1), (0, 0, 255), 3)
    mark_blue_rectangle = cv2.rectangle(image_cpy, (x_min2, y_min2), (x_min2 + w_min2, y_min2 + h_min2), (255, 0, 0), 3)
    mark_green_rectangle = cv2.rectangle(image_cpy, (x_min, y_min), (x_min + w_min, y_min + h_min), (0, 255, 0), 3)
    # cv2.imshow("Final", image_cpy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    text_box = [x_min, y_min, x_min+w_min, y_min+h_min]
    #print (text_box)

    """
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_blur1.png', image_blur)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_laplace_bin_open2.png', laplacian)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_laplace_inv_bin_open2.png', laplacian_inv)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_laplace_wtht_saturated3.png', laplacian_wtht_sat)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_laplace_inv_wtht_saturated3.png', laplacian_inv_wtht_sat)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_laplace_close-open4.png', laplacian_open)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_find_box_laplace_inv_close-open4.png', laplacian_open_inv)"""
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

    bbox_output = [np.array([x_min, y_min]),np.array([x_min, y_min + h_min]),np.array([x_min + w_min, y_min + h_min]),np.array([x_min + w_min, y_min])]

    return text_box, text_mask, bbox_output






def find_boxes_canny2(image, f_name, printbox=False):

    image_cpy = image.copy()
    height, width, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # remove noise
    image_blur = cv2.GaussianBlur(image,(11,11),0)


    gray_image = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    gray_image_inv = 255 - gray_image


    #---- apply optimal Canny edge detection using the computed median----
    v = np.median(gray_image)
    sigma = 0.8

    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))

    v_i = np.median(gray_image_inv)

    lower_thresh_i = int(max(0, (1.0 - sigma) * v_i))
    upper_thresh_i = int(min(255, (1.0 + sigma) * v_i))



    # convolute with proper kernels
    canny = cv2.Canny(gray_image,lower_thresh,upper_thresh,apertureSize=3)
    size_thresh_lapl = 50
    th, canny = cv2.threshold(canny, size_thresh_lapl,255,cv2.THRESH_BINARY)

    canny_i = cv2.Canny(gray_image_inv,lower_thresh_i,upper_thresh_i,apertureSize=3)

    size_thresh_lapl = 50
    th, canny_i = cv2.threshold(canny_i, size_thresh_lapl,255,cv2.THRESH_BINARY)

    '''cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows'''

    #PRUEBA MARCOS MIRAR LINEAS HORIZONTALES Y VERTICALES
    ''' minLineLength=80
    lines = cv2.HoughLinesP(image=canny,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=30)
    #lines = cv2.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)

    text_mask = np.zeros((height, width), dtype=np.uint8)
    a,b,c = lines.shape
    for i in range(a):
        if lines[i][0][1] != lines[i][0][3] and lines[i][0][0] != lines[i][0][2]:
            continue
        if lines[i][0][1] < height*0.05 or lines[i][0][1] >  height - height*0.05:
            continue 
        if lines[i][0][0] < width*0.05 or lines[i][0][2] >  width - width*0.05:
            continue 
        cv2.line(text_mask, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 3, cv2.LINE_AA)
    
    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow('Canny',text_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    size_thresh_lapl = 10
    th, text_mask = cv2.threshold(text_mask, size_thresh_lapl,255,cv2.THRESH_BINARY)

    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_canny_horizontals.png', text_mask)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 100))
    text_maskClose = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, element)

    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow('Canny',text_maskClose)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_canny_final.png', text_maskClose)'''



    #Put the letters together

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    cannyClose = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, element)
    cannyClose_i = cv2.morphologyEx(canny_i, cv2.MORPH_CLOSE, element)


    # Discard the largest connected object, if it is too big (due to the painting drawing lines)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cannyClose, connectivity=4)
    
    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(nb_components)], key=lambda x: x[1])

    if(max_size > height*width*0.35):
        cannyClose[output == max_label] = 0    

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cannyClose_i, connectivity=4)
    
    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(nb_components)], key=lambda x: x[1])

    if(max_size > height*width*0.35):
        cannyClose_i[output == max_label] = 0    


    #remove noise

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,11))
    cannyOpen = cv2.morphologyEx(cannyClose, cv2.MORPH_OPEN, element)
    cannyOpen_i = cv2.morphologyEx(cannyClose_i, cv2.MORPH_OPEN, element)



    #Put text together

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image.shape[1]*0.1), 5))
    cannyClose2 = cv2.morphologyEx(cannyOpen, cv2.MORPH_CLOSE, element)
    cannyClose2_i = cv2.morphologyEx(cannyOpen_i, cv2.MORPH_CLOSE, element)

    #remove noise

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cannyOpen2 = cv2.morphologyEx(cannyClose2, cv2.MORPH_OPEN, element)
    cannyOpen2_i = cv2.morphologyEx(cannyClose2_i, cv2.MORPH_OPEN, element)

    """
    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",cannyClose)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",cannyOpen)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",cannyClose2)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",cannyOpen2)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",canny_i)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",cannyClose_i)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",cannyOpen_i)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",cannyClose2_i)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",cannyOpen2_i)
    cv2.waitKey(0)
    cv2.destroyAllWindows """

    contours_n, hierarchy = cv2.findContours(cannyOpen2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours_i, hierarchy = cv2.findContours(cannyOpen2_i, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_n + contours_i

    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_canny_final.png', cannyOpen2)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_canny_i_final.png', cannyOpen2_i)

    #print(len(contours))
    
    image_cpy = image.copy()

    x_min1 = y_min1 = w_min1 = h_min1 = 0
    x_min2 = y_min2 = w_min2 = h_min2 = 0

    x_min = y_min = w_min = h_min = 0
    best_rectangles = []

    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        mark_brown_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (0, 100, 100), 3)
        if w < int(width * 0.05) or w > int(width * 0.75) or h < int(height * 0.01) or h > int(height*0.5):
            continue
        if h > w*0.55:
            continue
        if h < w*0.05:
            continue

        mark_purple_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (100, 0, 100), 3)

        if (x + (w / 2.0) < (width /2.0) - width * 0.08) or (x + (w / 2.0) > (width / 2.0) + width * 0.08):
            continue

        mark_red_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (0, 0, 255), 3)

        if y + h/2 > height*0.40 and y + h/2 < (height*0.6):
            continue
        mark_blue_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (255, 0, 0), 3)
    
        if h*w > width*height*0.15:
            continue

        mark_white_rectangle = cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (255, 255, 255), 3)

        """if h*w > h_min*w_min:
            x_min = x
            y_min = y
            w_min = w
            h_min = h"""

        best_rectangles.append(cnt)
        
    maxScore = 100000000


    for idx, cnt in enumerate(best_rectangles):
        x, y, w, h = cv2.boundingRect(cnt)
        dist_centre_x =  abs((x + (w / 2.0)) -  (width /2.0))
        dist_centre_y =  abs((y + (h / 2.0)) -  (height /2.0))
        diag = np.sqrt(w**2+h**2)
        #print(dist_centre_x)
        #print(dist_centre_y)
        #print(diag)

        #lower score the better --> less distance to the centre x , biggest diag and biggest dist from the center y
        #score = dist_centre_x/(diag + dist_centre_y)
        score = dist_centre_x/(diag + dist_centre_y)
        #print(score)

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

    """cv2.namedWindow("Canny", cv2.WINDOW_NORMAL) 
    cv2.imshow("Canny",image_cpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    


    text_box = [x_min, y_min, x_min+w_min, y_min+h_min]
    #print (text_box)


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

    bbox_output = [np.array([x_min, y_min]),np.array([x_min, y_min + h_min]),np.array([x_min + w_min, y_min + h_min]),np.array([x_min + w_min, y_min])]

    return text_box, text_mask, bbox_output
