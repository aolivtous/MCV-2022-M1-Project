from turtle import window_height
from main import *

base_dir = '../'
name_query='qsd1_w2/'
directory_query = base_dir + name_query
directory_output = base_dir + 'boxes/'
#from scipy.signal import find_peaks

def prova1(base_dir,directory_query,name_query,directory_output):
    for filename in os.scandir(directory_query):
        f = os.path.join(directory_query, filename)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0]
            image = cv2.imread(f,0)
            '''
            #prova per mirar si partint la imatge en dos es mes facil. Problema: Com sabem en  quina meitat esta la box
            image_up = image[:int(len(image[:][0])/2),:]
            image_down = image[int(len(image[:][0])/2):,:]

            cv2.imshow('',image_down)
            cv2.waitKey(0)

            #et diu quines son les tres intensitats mes presents, ja que usualment la box te la mateixa intensitat de fons, i t'ho segmenta en funcio d'aquestes intensitat
            #problema: Si el quadre té una intensitat molt present, serà aquesta la predominant i no la de la boxa, pero sino funciona prou bé
            histr = cv2.calcHist([image],[0],None,[256],[0,256])
            plt.plot(histr)
            plt.show()
            histr2 = cv2.calcHist([image_up],[0],None,[256],[0,256])
            plt.plot(histr2)
            plt.show()'''

            histr = cv2.calcHist([image],[0],None,[256],[0,256])
            sorted_hist = sorted(histr,reverse = True)[:1]
            indexes = []
            histr_list=list(histr)
            for i in range(len(sorted_hist)):
                indexes.append(histr_list.index(int(sorted_hist[i])))

            #ara mateix hi ha una approach que consisteix en aprofitar que la box tindra alguna linea (part sense lletres) on hi haura el mateix pixel repetit moltes vegades, pero no acaba de funcionar
            M,N = image.shape

            #kernel = np.ones((2,2),np.uint8)
            #image_e= cv2.dilate(image,kernel,iterations = 5)
            image_th = np.zeros((M,N))
            for i in range(M-1):
    
                for j in range(N-1):
                    '''if image[i][j] == image[i][j+30] and image[i][j] == image[i+10][j]:# and image_e[i][j] == image_e[i-1][j] and image_e[i][j] == image_e[i][j-1]:
                        image_th[i][j] = 255
                    else:
                        image_th[i][j]=0'''
                     #fer això si es vol l'approach de la intensitat maxima
                    if image[i][j] in indexes:
                        image_th[i][j] = 255
                    else: 
                        image_th[i][j] = 0

            th, image_th = cv2.threshold(image_th, 128, 255, cv2.THRESH_BINARY)
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            mask_close = cv2.morphologyEx(image_th, cv2.MORPH_OPEN, element, iterations=2)
            
            try:
                os.makedirs(directory_output)
            except FileExistsError:
                # Directory already exists
                pass

            cv2.imwrite(directory_output + f_name + '.jpg', image)
            #cv2.imwrite(directory_output + f_name + '__eroded.jpg',image_th)
            cv2.imwrite(directory_output  + f_name + '_box'+ '.jpg', mask_close)
            
    return





import pickle
with open(directory_query+'text_boxes.pkl', 'rb') as f:
    data = pickle.load(f)


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



def prova2(base_dir,directory_query,name_query,directory_output):
    for filename in os.scandir(directory_query):
        f = os.path.join(directory_query, filename)
        # checking if it is a file
        if f.endswith('07.jpg'):
            f_name = filename.name.split('.')[0]
            image = cv2.imread(f)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            l,a,b = cv2.split(image_lab)
            '''
            cv2.imshow('',l)
            cv2.waitKey(0)
            cv2.imshow('',a)
            cv2.waitKey(0)
            cv2.imshow('',b)
            cv2.waitKey(0)'''

            ret, maska = cv2.threshold(a, 0 ,255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
            ret, maskb = cv2.threshold(b, 0 ,255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
            ret, maskg = cv2.threshold(image_gray, 0 ,255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            ret, maskl = cv2.threshold(a, 0 ,255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)


            
            # cv2.imshow('mask a',maska)
            # cv2.waitKey(0)
            cv2.imshow('mask b',maskg)
            cv2.waitKey(0)
            # cv2.imshow('mask gray',maskg)
            # cv2.waitKey(0)
            height,width,channels = image.shape

            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            top_hat_x = cv2.morphologyEx(maskg, cv2.MORPH_TOPHAT, element, iterations=5)
            black_hat_x = cv2.morphologyEx(maskg, cv2.MORPH_BLACKHAT, element, iterations=5)
            '''rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))
            top_hat = cv2.dilate(maska, rect_kernel, iterations = 1)'''
            total_hat = top_hat_x + black_hat_x
            cv2.imshow('top_hat x',top_hat_x)
            cv2.waitKey(0)
            cv2.imshow('black_hat x',black_hat_x)
            cv2.waitKey(0)
            cv2.imshow('total hat ',total_hat)
            cv2.waitKey(0)

            
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            close_total = cv2.morphologyEx(total_hat, cv2.MORPH_CLOSE, element)
            cv2.imshow('close total hat ',close_total)
            cv2.waitKey(0)

            ''''
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2,20))
            top_hat_y = cv2.morphologyEx(maska, cv2.MORPH_TOPHAT, element)
            black_hat_y = cv2.morphologyEx(maska, cv2.MORPH_BLACKHAT, element)
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))
            top_hat = cv2.dilate(maska, rect_kernel, iterations = 1)

            cv2.imshow('top_hat y',top_hat_y)
            cv2.waitKey(0)
            cv2.imshow('black_hat y',black_hat_y)
            cv2.waitKey(0)

            black_hat_intersect = cv2.bitwise_and(top_hat_y,top_hat_x)
            cv2.imshow('black_hat intersect',black_hat_intersect)
            cv2.waitKey(0)'''

            # element = cv2.getStructuringElement(cv2.MORPH_RECT, (10,2))
            # close_top_hat_x = cv2.morphologyEx(top_hat_x, cv2.MORPH_CLOSE, element)
            # close_black_hat_x = cv2.morphologyEx(black_hat_x, cv2.MORPH_CLOSE, element)
            # open_top_hat_x = cv2.morphologyEx(top_hat_x, cv2.MORPH_OPEN, element)
            # open_black_hat_x = cv2.morphologyEx(black_hat_x, cv2.MORPH_OPEN, element)

            # cv2.imshow('close top_hat x',close_top_hat_x)
            # cv2.waitKey(0)
            # cv2.imshow('close black_hat x',close_black_hat_x)
            # cv2.waitKey(0)
            # cv2.imshow('open top_hat x',open_top_hat_x)
            # cv2.waitKey(0)
            # cv2.imshow('open black_hat x',open_black_hat_x)
            # cv2.waitKey(0)

            retr_mode = cv2.RETR_EXTERNAL
            contours_top, hierarchy_top = cv2.findContours(top_hat_x, retr_mode,cv2.CHAIN_APPROX_SIMPLE)
            contours_black, hierarchy_top = cv2.findContours(black_hat_x, retr_mode,cv2.CHAIN_APPROX_SIMPLE)
            contours_total, hierarchy_total = cv2.findContours(total_hat, retr_mode,cv2.CHAIN_APPROX_SIMPLE)
            contours_total_close, hierarchy_total = cv2.findContours(close_total, retr_mode,cv2.CHAIN_APPROX_SIMPLE)
            # contours_close_top, hierarchy_top = cv2.findContours(close_top_hat_x, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            # contours_close_black, hierarchy_top = cv2.findContours(close_black_hat_x, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            # contours_open_top, hierarchy_top = cv2.findContours(open_top_hat_x, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            # contours_open_black, hierarchy_top = cv2.findContours(open_black_hat_x, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            
            im2 = image.copy()
            x_min = 0
            y_min = 0
            w_min = 0
            h_min = 0
            diffmin = 1000000
            
            # Angel change
            image_rgb_cpy = image.copy()
            best_x, best_y, best_w, best_h, b_bl_cnt, b_tl_cnt, b_br_cnt, b_tr_cnt, min_var = 0, 0, 0, 0, 0, 0, 0, 0, 99999

            contours = contours_top + contours_black + contours_total#+ contours_close_top + contours_close_black + contours_open_top + contours_open_black
            contours = contours_total + contours_total_close

            for idx, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                if w < int(width*0.99) and w > int(width*0.02) and h < int(height*0.4) and h > int(height*0.02):
                    # maxmin = l[y:y + h, x:x + w].max() - l[y:y + h, x:x + w].min()
                    # var = np.var(maska[y:y + h, x:x + w])
                    max_val = image_rgb[y:y + h, x:x + w].max()
                    min_val = image_rgb[y:y + h, x:x + w].min()
                    
                    print('x', x, 'y', y, 'w', w, 'h', h)

                    bl_cnt = image_rgb[y, x]
                    tl_cnt = image_rgb[y + h - 1, x]
                    br_cnt = image_rgb[y, x + w - 1]
                    tr_cnt = image_rgb[y + h - 1, x + w - 1]
                    
                    
                    # print('color bl:', bl_cnt)
                    # print('color tl:', tl_cnt)
                    # print('color br:', br_cnt)
                    # print('color tr:', tr_cnt)

                    color_var = np.var([bl_cnt, tl_cnt, br_cnt, tr_cnt], axis=0)
                    print('color var:', color_var)
                    mean_var = np.mean(color_var)
                    print('mean var:', mean_var)

                    mark_red_rectangle = cv2.rectangle(image_rgb_cpy, (x, y), (x + w, y + h), (0, 0, 255), 3)

                    if idx == 0 or mean_var < min_var:
                        best_x = x
                        best_y = y
                        best_w = w
                        best_h = h
                        b_bl_cnt = bl_cnt
                        b_tl_cnt = tl_cnt
                        b_br_cnt = br_cnt
                        b_tr_cnt = tr_cnt
                        min_var = mean_var
                        print('changing mean var:', mean_var)
                        print('changing color bl:', bl_cnt)
                        print('changing color tl:', tl_cnt)
                        print('changing color br:', br_cnt)
                        print('changing color tr:', tr_cnt)

            print('final mean var:', min_var)
            print('final color bl:', b_bl_cnt)
            print('final color tl:', b_tl_cnt)
            print('final color br:', b_br_cnt)
            print('final color tr:', b_tr_cnt)
            mark_green_rectangle = cv2.rectangle(image_rgb_cpy, (best_x, best_y), (best_x + best_w, best_y + best_h), (0, 255, 0), 2)

                # if w < width-int(width*0.05) and w > int(width*0.15) and h < int(height*0.3) and h > int(height*0.03):
                #     # Drawing a rectangle on copied image
                #     #print(var)
                #     #print(maxmin)
                #     rectangle = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 3)

                #     cropped = image_rgb[y:y + h, x:x + w]
                #     diff = color_factor(cropped)
                #     print(f'diff={diff}')

                #     if diff < diffmin:
                #         diffmin=diff
                #         x_min = x
                #         y_min = y
                #         w_min = w
                #         h_min = h

                        
            #Extendre el rectangle escollit Verd per tenir la box sencera
            # 
            #
            # 
            # 
            # Evaluar les dues boxes amb bbox_iou()           
            # print(f'Imatge {f_name} : diffmin={diffmin}')
            # rectminvar = cv2.rectangle(image_rgb_cpy, (x_min, y_min), (x_min + w_min, y_min + h_min), (0, 255, 0), 2)
            # # Cropping the text block for giving input to OCR
            # cropped = image[y_min:y_min + h_min, x_min:x_min + w_min]
            cv2.namedWindow('cropped', cv2.WINDOW_NORMAL)
            cv2.imshow('cropped',image_rgb_cpy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return

def prova3(base_dir,directory_query,name_query,directory_output):
    for filename in os.scandir(directory_query):
        f = os.path.join(directory_query, filename)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0]
            image = cv2.imread(f)
            height, width, channels = image.shape
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            cv2.imshow('grayscale', image_gray)
            cv2.waitKey(0)

            # _, mask = cv2.threshold(image_gray, 180, 255, cv2.THRESH_OTSU) # ! Comprovar 180 i veure si es pot canbiar el thresh binary per cv2.THRESH_OTSU
            # image_bit = cv2.bitwise_and(image_gray, image_gray, mask=mask)
            # cv2.imshow('image bit bin thresh I', image_bit)
            # cv2.waitKey(0)

            _, image_bit = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # for black text , cv.THRESH_BINARY_INV
            cv2.imshow('image bit bin thresh II', image_bit)
            cv2.waitKey(0)

            element = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # ! provar altres formes to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
            image_dil = cv2.dilate(image_bit, element, iterations=1) # ! dilate , more the iteration more the dilation, provar altres iteracions
            cv2.imshow('image bit dilated', image_dil)
            cv2.waitKey(0)
            
            contours, _ = cv2.findContours(image_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )  # ! findContours returns 3 variables for getting contours provar algo diferent pel retr_external
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # If it's that small then it's not text
                if w < int(width * 0.01) or h < int(height * 0.01):
                    continue

                # Rectangle drawing of the contour
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.imshow('text_detection_result', image)
            cv2.waitKey()
            cv2.destroyAllWindows()

def prova4(base_dir,directory_query,name_query,directory_output):
    for filename in os.scandir(directory_query):
        f = os.path.join(directory_query, filename)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0]
            image = cv2.imread(f)
            height, width, channels = image.shape
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            bin_image = np.zeros((height, width), dtype=np.uint8)

            image_th = np.zeros((height, width))
            for i in range(height - 1):
                for j in range(width - 1):
                    if image_hsv[i][j][1] * (1/2.55) > 10 or (image_hsv[i][j][2] * (1/2.55) > 25 and image_hsv[i][j][2] * (1/2.55) < 75): # ! Provar sense or
                        bin_image[i][j] = 255
                    else:
                        bin_image[i][j] = 0

            cv2.imwrite(directory_output + f_name + '_bin.png', bin_image)

# prova2(base_dir,directory_query,name_query,directory_output)
# prova3(base_dir,directory_query,name_query,directory_output)
# prova1(base_dir,directory_query,name_query,directory_output)
prova4(base_dir,directory_query,name_query,directory_output)
