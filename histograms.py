from main import *

class histograms: 
    def __init__(self, hist_ch1, hist_ch2, hist_ch3): 
        self.hist_ch1 = hist_ch1
        self.hist_ch2 = hist_ch2
        self.hist_ch3 = hist_ch3

def get_histograms(directory, colorSpace, notQuery, withMask):
    """
    It takes a directory, a color space, and a boolean as input and returns a dictionary of histograms
    
    :param directory: The directory where the images are stored
    :param colorSpace: The color space to use for the histogram
    :param notQuery: If the image is a query image or not
    :return: A dictionary of histograms.
    """
    hist_dict = {}
    for filename in os.scandir(directory):
        f = os.path.join(directory, filename)
        
        # checking if it is a file
        if f.endswith('.jpg'):
            # Splitting the file name and getting the file name without the extension.
            split_f = f.split('/')[-1]
            f_name = split_f.split('.')[0]

            if notQuery:
                f_name = f_name.split('_')[1]
            else:
                f_name = f_name[-5:]


            image = cv2.imread(f)

            if colorSpace == "HSV":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)           
            elif colorSpace == "LAB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)          
            elif colorSpace == "YCrCb":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            elif colorSpace == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if withMask:    
                mask_name = os.path.join(directory, "predicted_masks/" + f_name + ".png")
                mask = cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE)                
                hist_ch1 = cv2.calcHist([image], [0], mask, [256], [0, 255])
                hist_ch2 = cv2.calcHist([image], [1], mask, [256], [0, 255])
                hist_ch3 = cv2.calcHist([image], [2], mask, [256], [0, 255])

            
            else:

                hist_ch1 = cv2.calcHist([image], [0], None, [256], [0, 255])
                hist_ch2 = cv2.calcHist([image], [1], None, [256], [0, 255])
                hist_ch3 = cv2.calcHist([image], [2], None, [256], [0, 255])

            hist_ch1 /= hist_ch1.sum()
            hist_ch2 /= hist_ch2.sum()
            hist_ch3 /= hist_ch3.sum()
            
            hist_dict[f_name] = (histograms(hist_ch1, hist_ch2, hist_ch3))
    return hist_dict