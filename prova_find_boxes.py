from main import *

base_dir = '../'
name_query='qsd1_w2/'
directory_query = base_dir + name_query
directory_output = base_dir + 'boxes/'
#from scipy.signal import find_peaks
import pandas as pd


def prova1(base_dir,directory_query,name_query,directory_output):
    for filename in os.scandir(directory_query):
        f = os.path.join(directory_query, filename)
        # checking if it is a file
        if f.endswith('00022.jpg'):
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
            plt.show()

            sorted_hist = sorted(histr,reverse = True)[:3]
            indexes = []
            histr_list=list(histr)
            for i in range(len(sorted_hist)):
                indexes.append(histr_list.index(int(sorted_hist[i])))'''

            #ara mateix hi ha una approach que consisteix en aprofitar que la box tindra alguna linea (part sense lletres) on hi haura el mateix pixel repetit moltes vegades, pero no acaba de funcionar
            M,N = image.shape

            #kernel = np.ones((2,2),np.uint8)
            #image_e= cv2.dilate(image,kernel,iterations = 5)
            image_th = np.zeros((M,N))
            for i in range(M-11):
    
                for j in range(N-31):
                    if image[i][j] == image[i][j+30] and image[i][j] == image[i+10][j]:# and image_e[i][j] == image_e[i-1][j] and image_e[i][j] == image_e[i][j-1]:
                        image_th[i][j] = 255
                    else:
                        image_th[i][j]=0
                    ''' #fer això si es vol l'approach de la intensitat maxima
                    if image[i][j] in indexes:
                        image_th[i][j] = 255
                    else: 
                        image_th[i][j] = 0'''

            th, image_th = cv2.threshold(image_th, 128, 255, cv2.THRESH_BINARY)
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            mask_close = cv2.morphologyEx(image_th, cv2.MORPH_OPEN, element, iterations=2)
            
            cv2.imwrite(directory_output + f_name + '.jpg',image)
            cv2.imwrite(directory_output + f_name + '__eroded.jpg',image)
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
        if f.endswith('00022.jpg'):
            f_name = filename.name.split('.')[0]
            image = cv2.imread(f)
            image_rgb = cv2.imread(f)

            cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

            cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
            l,a,b = cv2.split(image)
            '''
            cv2.imshow('',l)
            cv2.waitKey(0)
            cv2.imshow('',a)
            cv2.waitKey(0)
            cv2.imshow('',b)
            cv2.waitKey(0)'''

            ret, maska = cv2.threshold(a, 0 ,255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
            ret, maskb = cv2.threshold(b, 0 ,255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
            
            cv2.imshow('',maska)
            cv2.waitKey(0)
            cv2.imshow('',maskb)
            cv2.waitKey(0)

            height,width,channels = image.shape

            element = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.10),2))
            top_hat_x = cv2.morphologyEx(maska, cv2.MORPH_TOPHAT, element, iterations=1)
            black_hat_x = cv2.morphologyEx(maska, cv2.MORPH_BLACKHAT, element, iterations=1)
            '''rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))
            top_hat = cv2.dilate(maska, rect_kernel, iterations = 1)'''
            
            cv2.imshow('top_hat x',top_hat_x)
            cv2.waitKey(0)
            cv2.imshow('black_hat x',black_hat_x)
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

            element = cv2.getStructuringElement(cv2.MORPH_RECT, (10,2))
            close_top_hat_x = cv2.morphologyEx(top_hat_x, cv2.MORPH_CLOSE, element)
            close_black_hat_x = cv2.morphologyEx(black_hat_x, cv2.MORPH_CLOSE, element)
            
            cv2.imshow('close top_hat x',close_top_hat_x)
            cv2.waitKey(0)
            cv2.imshow('close black_hat x',close_black_hat_x)
            cv2.waitKey(0)

            contours_top, hierarchy_top = cv2.findContours(top_hat_x, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            contours_black, hierarchy_top = cv2.findContours(black_hat_x, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            contours_close_top, hierarchy_top = cv2.findContours(close_top_hat_x, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            contours_close_black, hierarchy_top = cv2.findContours(close_black_hat_x, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

            im2 = image.copy()
            x_min = 0
            y_min = 0
            w_min = 0
            h_min = 0
            diffmin = 1000000
            
            contours = contours_top + contours_black + contours_close_top + contours_close_black
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                maxmin = l[y:y + h, x:x + w].max() - l[y:y + h, x:x + w].min()
                var = np.var(maska[y:y + h, x:x + w])

                if w < width-int(width*0.05) and w > int(width*0.15) and h < int(height*0.3) and h > int(height*0.03):
                    # Drawing a rectangle on copied image
                    #print(var)
                    #print(maxmin)
                    rectangle = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 3)

                    cropped = image_rgb[y:y + h, x:x + w]
                    diff = color_factor(cropped)
                    print(f'diff={diff}')

                    if diff < diffmin:
                        diffmin=diff
                        x_min = x
                        y_min = y
                        w_min = w
                        h_min = h
                        
            #Extendre el rectangle escollit Verd per tenir la box sencera
            # 
            #
            # 
            # 
            # Evaluar les dues boxes amb bbox_iou()           
            print(f'Imatge {f_name} : diffmin={diffmin}')
            rectminvar = cv2.rectangle(im2, (x_min, y_min), (x_min + w_min, y_min + h_min), (0, 255, 0), 2)
            # Cropping the text block for giving input to OCR
            cropped = image[y_min:y_min + h_min, x_min:x_min + w_min]
            cv2.imshow('cropped',im2)
            cv2.waitKey(0)
 
            cv2.destroyAllWindows()

            
    return

prova2(base_dir,directory_query,name_query,directory_output)
