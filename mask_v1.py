from main import *

def generate_masks(dir_query2, dir_output, threshold_value=95, plot_histograms = False):
    """
    > The function takes as input the path of the image and the path of the output directory. It returns
    the mask of the image
    
    :param path_image_query2: path to the image to be processed
    :param dir_output: the directory where the output images will be saved
    :param threshold_value: The value that the pixels will be compared to, defaults to 100 (optional)
    :return: the mask of the image.
    """
    for filename in os.scandir(dir_query2):
        f = os.path.join(dir_query2, filename)
        # checking if it is a file
        if f.endswith('.jpg'): #and f.endswith('00001.jpg'):
            # Splitting the file name and getting the file name without the extension.
            filename = filename.name
            image = cv2.imread(f)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            kernel = np.ones((5,5),np.uint8)
            image_gray = cv2.erode(image_gray,kernel,iterations = 2)
    
            hist_gray = cv2.calcHist([image_gray], [0], None, [256], [0,256])
            # cv2.normalize(hist_gray, hist_gray, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
            hist_gray /= hist_gray.sum()

            if plot_histograms:
                plt.plot(hist_gray)
                plt.savefig(dir_output + '/hist_' + filename)
                plt.clf() 


            # Apply threshold and show the image
            ret,imgt = cv2.threshold(image_gray,threshold_value,255,cv2.THRESH_BINARY_INV)

            # Apply close morphology operator in order denoise
            dilation = 5
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilation+1, 2*dilation+1),(int(dilation/2), int(dilation/2)))
            mask_close = cv2.morphologyEx(imgt, cv2.MORPH_CLOSE, element, iterations=10)
            
            # Save the image in dir_output
            no_ext_f = filename.split('.')[0]
            filename =  dir_output + '/' + no_ext_f + '.png'
            cv2.imwrite(filename, mask_close)
            
    return


