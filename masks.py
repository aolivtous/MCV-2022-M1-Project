import global_variables
import cv2
import numpy as np

def generate_masks_otsu(image, f_name, splitimage):
    """
    > The function takes as input the path of the image and the path of the output directory. It returns
    the mask of the image
    
    :param path_image_query2: path to the image to be processed
    :param dir_output: the directory where the output images will be saved
    :param threshold_value: The value that the pixels will be compared to, defaults to 100 (optional)
    :return: the mask of the image.
    """

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
    kernel = np.ones((2,2),np.uint8)
    image_e= cv2.erode(image_gray,kernel,iterations = 2)
    a,imgt = cv2.threshold(image_e, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
    # Apply close morphology operator in order denoise
    size = 5
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*size+1, 2*size+1))
    mask_close = cv2.morphologyEx(imgt, cv2.MORPH_CLOSE, element, iterations=5)
                        
    # Save the image in dir_results
    cv2.imwrite(f'{global_variables.dir_results}{f_name}.png', mask_close)
            
    return

def generate_masks(image, f_name, splitimage): #NOVA FUNCIO PER DETECTAR ELS QUADRES fent el LAPLACIA
    """
    > The function takes as input the path of the image and the path of the output directory. It returns
    the mask of the image
    
    :param path_image_query2: path to the image to be processed
    :param dir_output: the directory where the output images will be saved
    :param threshold_value: The value that the pixels will be compared to, defaults to 100 (optional)
    :return: the mask of the image.
    """
    image_cpy = image.copy()
    width, height, _ = image.shape
    print(width)
    print(height)
    

    # remove noise
    image = cv2.GaussianBlur(image,(7,7),0)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # convolute with proper kernels
    laplacian = cv2.Laplacian(gray_image,cv2.CV_64F, ksize = 3)

    cv2.imshow("Laplacian", laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Apply close morphology operator in order denoise
  
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_open = cv2.morphologyEx(laplacian, cv2.MORPH_OPEN, element)

    element_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(mask_open,element_dil)

    cv2.imshow("Laplacian", mask_open)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    cv2.imshow("Laplacian", dilation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    mask_open2 = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, element2)

    cv2.imshow("Laplacian", mask_open2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    contours, hierarchy = cv2.findContours(np.uint8(mask_open2), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

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

    if len(contours) > 1 :
        secondlargestcontour = sorteddata[1][1]
        x2, y2, w2, h2 = cv2.boundingRect(secondlargestcontour)

        if(w2*h2 > 0.1*width*height):
            mark_red_rectangle = cv2.rectangle(image_cpy, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 3)

      
        
    cv2.imshow("Laplacian", image_cpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


 