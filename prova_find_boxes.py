from main import *

base_dir = '../'
name_query='qsd1_w2/'
directory_query = base_dir + name_query
directory_output = base_dir + 'boxes/'
from scipy.signal import find_peaks
import pandas as pd



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
        

