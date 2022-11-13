import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def callback(x):
    print(x)

# !
test_image = '16'
name_query = 'qsd1_w5'

# Constant arguments
name_db = 'BBDD'
dir_base = '../../'
results_name = 'results'
aux_name = 'aux'
# Directories assignment (always end with /)
dir_db = f'{dir_base}{name_db}/' 
dir_query = f'{dir_base}{name_query}/'

f = f'{dir_query}000{test_image}.jpg'
img = cv2.imread(f, 0) #read image as grayscale

canny = cv2.Canny(img, 85, 255) 

cv2.namedWindow('image') # make a window with name 'image'
# Set window to be resizable
cv2.resizeWindow('image', 600,600)
cv2.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image

while(1):
    numpy_horizontal_concat = np.concatenate((img, canny), axis=1) # to display image side by side
    cv2.imshow('image', numpy_horizontal_concat)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: #escape key
        break
    l = cv2.getTrackbarPos('L', 'image')
    u = cv2.getTrackbarPos('U', 'image')

    canny = cv2.Canny(img, l, u)

cv2.destroyAllWindows()