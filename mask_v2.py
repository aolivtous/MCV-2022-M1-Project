from main import *
import main
import os


def generate_mask(path_image_query2, dir_output, threshold_value=140):
    """
    > The function takes as input the path of the image and the path of the output directory. It returns
    the mask of the image
    
    :param path_image_query2: path to the image to be processed
    :param dir_output: the directory where the output images will be saved
    :param threshold_value: The value that the pixels will be compared to, defaults to 100 (optional)
    :return: the mask of the image.
    """

    image = cv2.imread(path_image_query2) #00004
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist(image_gray, [0], None, [256], [0,256])
    cv2.normalize(hist_gray, hist_gray, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
    plt.plot(hist_gray)
    plt.xlim([0,256])
    
    # Find the minimum between the two maximums of histogram (valle)
    threshold_value = 150

    # Apply threshold and show the image
    ret,imgt = cv2.threshold(image_gray,threshold_value,255,cv2.THRESH_BINARY)
    cv2_imshow(imgt)
    
    # Reverse the image in order to have background --> black
    imgt=255-imgt
    cv2_imshow(imgt)

    # Apply opening morphology operator in order denoise
    dilation = 1
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilation+1, 2*dilation+1),
                                            (int(dilation/2), int(dilation/2)))
    mask_opening = cv2.morphologyEx(imgt, cv2.MORPH_OPEN, element, iterations=3)
    

    # Save the image in dir_output
    split_f = path_image_query2.split('/')[-1]
    filename =  dir_output + '/' + split_f
    cv2.imwrite(filename, mask_opening)

    return



