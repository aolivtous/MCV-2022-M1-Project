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


 