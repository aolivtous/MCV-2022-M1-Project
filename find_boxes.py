from turtle import window_height
from main import *


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


def find_boxes(directory_query,directory_output, printbox=False):
    bbox_output = []
    for filename in os.scandir(directory_query):
        f = os.path.join(directory_query, filename)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0]
            image = cv2.imread(f)
            height, width, channels = image.shape
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

            element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width/20.0), 1))
            bin_image_close = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, element)

            element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            bin_image_close = cv2.morphologyEx(bin_image_close, cv2.MORPH_OPEN, element)

            element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width/4.0), 1))
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
                print('diff', diff)
                print('x', x)
                print('y', y)
                if diff < mindiff:
                    mindiff=diff
                    x_min = x
                    y_min = y
                    w_min = w
                    h_min = h

            #Extensió del rectangle 
            tol = 10
            while abs(int(image_hsv[y_min + h_min, x_min][1]) - int(image_hsv[y_min + h_min +1, x_min][1])) < tol and (image_hsv[y_min + h_min, x_min][2]<50 or image_hsv[y_min + h_min, x_min][2]>200) :
                h_min = h_min + 1
            
            while abs(int(image_hsv[y_min + h_min, x_min][1]) - int(image_hsv[y_min + h_min, x_min - 1][1])) < tol and (image_hsv[y_min + h_min, x_min][2]<50 or image_hsv[y_min + h_min, x_min][2]>200) :
                x_min = x_min - 1

            while abs(int(image_hsv[y_min, x_min + w_min][1]) - int(image_hsv[y_min, x_min + w_min + 1][1])) < tol and (image_hsv[y_min, x_min + w_min][2]<50 or image_hsv[y_min, x_min + w_min][2]>200):
                w_min = w_min + 1
            
            while abs(int(image_hsv[y_min, x_min + w_min][1]) - int(image_hsv[y_min - 1 , x_min + w_min][1])) < tol and (image_hsv[y_min, x_min + w_min][2]<50 or image_hsv[y_min, x_min + w_min][2]>200):
                y_min = y_min - 1

                
            print(mindiff)
            mark_green_rectangle = cv2.rectangle(image_cpy, (x_min, y_min), (x_min + w_min, y_min + h_min), (0, 255, 0), 3)
            
            if printbox:
                cv2.imwrite(directory_output + f_name + '_bin.png', bin_image)
                cv2.imwrite(directory_output + f_name + '_bin_close.png', bin_image_close)
                cv2.imwrite(directory_output + f_name + '_bin_close_close.png', bin_image_close_close)
                cv2.imwrite(directory_output + f_name + '_rectangle_exten.png', image_cpy)
        
        bbox_output.append([x_min, y_min, x_min+w_min, y_min+h_min])   
            
    return bbox_output



def bbox_iou(bboxA, bboxB):
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


def find_boxes_eval(list_bbox_prediction, list_bbox_solution):
    iou_list=[]
    for i in range(len(list_bbox_prediction)):
        iou = bbox_iou(list_bbox_prediction[i], list_bbox_solution[i][0])
        iou_list.append(iou)


base_dir = '../'
name_query='qsd1_w2/'
directory_query = base_dir + name_query
directory_output = base_dir + 'boxes/'

with open(directory_query+'text_boxes.pkl', 'rb') as f:
        data = pickle.load(f)
    
print(data)

    