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
        image_up = image[:int(len(image[:][0])/2),:]
        image_down = image[int(len(image[:][0])/2):,:]

        cv2.imshow('',image_down)
        cv2.waitKey(0)


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


        M,N = image.shape

 
        image_th = np.zeros((M,N))
        for i in range(M-1):
   
            for j in range(N-1):
                if image[i][j] == image[i][j+1] and image[i][j] == image[i+1][j] and image[i][j] == image[i-1][j] and image[i][j] == image[i][j-1]:
                    image_th[i][j] = 255
                else:
                    image_th[i][j]=0
                '''if image[i][j] in indexes:
                    image_th[i][j] = 255
                else: 
                    image_th[i][j] = 0'''

        th, image_th = cv2.threshold(image_th, 128, 255, cv2.THRESH_BINARY)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        #mask_close = cv2.morphologyEx(image_th, cv2.MORPH_OPEN, element, iterations=8)
        
        cv2.imwrite(directory_output + f_name + '.jpg',image)
        cv2.imwrite(directory_output  + f_name + '_box'+ '.jpg', image_th)
        

