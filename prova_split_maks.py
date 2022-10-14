from main import *

# Default arguments
name_bbdd = 'BBDD'
name_query = 'provamask'
method_search = 1
color_code = "LAB" # ["RGB", "HSV", "LAB", "YCrCb"]
distance_type = 'hellin' # Possible arguments of distance_type at argument_relations
backgrounds = True
boundingbox = True
solutions = True
plot_histograms = False
default_threshold = 95

    # Global variable
base_dir = "../"
output_name = "predictions"

   
    # Directories assignment
directory_bbdd = base_dir + name_bbdd
directory_query = base_dir + name_query
output_path = "/" + output_name + "/"
directory_output = directory_query + output_path
directory_proves = base_dir + output_path



def generate_masks(dir_query2, dir_output,plot_histograms = False):
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
            image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            l,a,b = cv2.split(image_lab)
            cv2.imshow("l image", l)
            cv2.waitKey(0)
            cv2.imshow("a image", a)
            cv2.waitKey(0)
            cv2.imshow("b image", b)
            cv2.waitKey(0)


            # Apply threshold and show the image
            mask_l = cv2.adaptiveThreshold(l,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,10)
            mask_a = cv2.adaptiveThreshold(a,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,10)
            mask_b = cv2.adaptiveThreshold(b,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,10)

            mask_l = 255 - mask_l
            mask_a = 255 - mask_a
            mask_b = 255 - mask_b

            cv2.imshow("l mask", mask_l)
            cv2.waitKey(0)
            cv2.imshow("a mask", mask_a)
            cv2.waitKey(0)
            cv2.imshow("b mask", mask_b)
            cv2.waitKey(0)

            # Apply opening morphology operator in order denoise
            size = 5
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*size+1, 2*size+1))
            mask_l = cv2.morphologyEx(mask_l, cv2.MORPH_CLOSE, element, iterations=10)
            mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_CLOSE, element, iterations=10)
            mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, element, iterations=10)

            cv2.imshow("l mask", mask_l)
            cv2.waitKey(0)
            cv2.imshow("a mask", mask_a)
            cv2.waitKey(0)
            cv2.imshow("b mask", mask_b)
            cv2.waitKey(0)
            
            # Save the image in dir_output
            no_ext_f = filename.split('.')[0]
            filename =  dir_output + '/' + no_ext_f + '.png'
            cv2.imwrite(filename, mask_a)
            
    return



def generate_masks2(dir_query2, dir_output,plot_histograms = False):
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

            cv2.imshow("image", image_gray)
            cv2.waitKey(0)

            blur = cv2.GaussianBlur(image_gray, (5,5),0)
            cv2.imshow("blur", blur)
            cv2.waitKey(0)
            ret, mask_close = cv2.threshold(blur, 0 ,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            mask_close = 255-mask_close 
            # Apply opening morphology operator in order denoise
            size = 5
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*size+1, 2*size+1))
            mask_close = cv2.morphologyEx(mask_close, cv2.MORPH_CLOSE, element, iterations=10)
            #mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_CLOSE, element, iterations=10)
            #mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, element, iterations=10)

            cv2.imshow("mask", mask_close)
            cv2.waitKey(0)
            
            # Save the image in dir_output
            no_ext_f = filename.split('.')[0]
            filename =  dir_output + '/' + no_ext_f + '.png'
            cv2.imwrite(filename, mask_close)
            
    return






generate_masks2(directory_query, directory_output, plot_histograms = False)