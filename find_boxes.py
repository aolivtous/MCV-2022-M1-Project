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

    """cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_text_laplacian.png', laplacian)
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_text_laplacian_open.png', np.uint8(mask_open))
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_text_laplaian_open_dilate.png', np.uint8(dilation))
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_text_laplaian_open_dilate_open.png', np.uint8(mask_open2))"""
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_text_laplacian_boxes.png', image_cpy)

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




# import cv2
# import pytesseract

# def find_box_ocr(img, f_name, printbox=False):

#     h, w, c = img.shape
#     boxes = pytesseract.image_to_boxes(img) 
#     for b in boxes.splitlines():
#         b = b.split(' ')
#         img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

#     cv2.imshow('img', img)
#     cv2.waitKey(0)
    
#     return 