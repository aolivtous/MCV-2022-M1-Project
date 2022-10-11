from main import *

class histograms: 
    def __init__(self, hist_ch1, hist_ch2, hist_ch3): 
        self.hist_ch1 = hist_ch1
        self.hist_ch2 = hist_ch2
        self.hist_ch3 = hist_ch3

"""def get_block_histograms(directory, output_name, blockLevel, bins, query, with_mask):
    
    Calculate and concatenate histograms made from parts of the image of a particular block level

    :param directory: The directory where the images are stored
    :param block level: [int] level 1: 1 histogram, level 2: 4 blocks --> 4 histograms concatenated, level 3: 16 blocks --> 16 histograms concatenated
    :param bins: number of bins of the histograms
    :param notQuery: If the image is a query image or not
    :return: A dictionary of histograms.
    
    hist_dict = {}
    for filename in os.scandir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0]
            if not query:
                f_name = f_name.split('_')[1]

            image = cv2.imread(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab) 

            #spliting the image into blocks
            
            n_patches = int((2**blockLevel)/2)
            
            M = image.shape[0]//n_patches
            N = image.shape[1]//n_patches

            tiles = [image[x:x+M,y:y+N] for x in range(0,image.shape[0]-image.shape[0]%n_patches,M) for y in range(0,image.shape[1]-image.shape[1]%n_patches,N)]
            concat_hist_ch1 = []
            concat_hist_ch2 = []
            concat_hist_ch3 = []

            concat_hist_ch1 = np.float32(concat_hist_ch1)
            concat_hist_ch2 = np.float32(concat_hist_ch2)
            concat_hist_ch3 = np.float32(concat_hist_ch3)

            if with_mask:  
                mask_name = os.path.join(directory, output_name + "/" + f_name + ".png")
                mask = cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE)

                mask_tiles = [mask[x:x+M,y:y+N] for x in range(0,mask.shape[0]-mask.shape[0]%n_patches,M) for y in range(0,mask.shape[1]-mask.shape[1]%n_patches,N)]

            for index,tile in enumerate(tiles):

                if with_mask:     
                    hist_ch1 = cv2.calcHist([tile], [0], mask_tiles[index], [bins], [0, 255])
                    hist_ch2 = cv2.calcHist([tile], [1], mask_tiles[index], [bins], [0, 255])
                    hist_ch3 = cv2.calcHist([tile], [2], mask_tiles[index], [bins], [0, 255])

                    plt.plot(hist_ch1)
                    plt.show()
                
                else:               
                    hist_ch1 = cv2.calcHist([tile], [0], None, [bins], [0, 255])
                    hist_ch2 = cv2.calcHist([tile], [1], None, [bins], [0, 255])
                    hist_ch3 = cv2.calcHist([tile], [2], None, [bins], [0, 255])

                if with_mask and mask_tiles[index]
                hist_ch1 /= hist_ch1.sum()
                hist_ch2 /= hist_ch2.sum()
                hist_ch3 /= hist_ch3.sum()  

                
                concat_hist_ch1 = np.append(concat_hist_ch1,hist_ch1)
                concat_hist_ch2 = np.append(concat_hist_ch2,hist_ch2)
                concat_hist_ch3 = np.append(concat_hist_ch3,hist_ch3)
            
            hist_dict[f_name] = (histograms(concat_hist_ch1, concat_hist_ch2, concat_hist_ch3))

    return hist_dict"""

def get_block_histograms(directory, output_name, blockLevel, bins, query):
    
    """Calculate and concatenate histograms made from parts of the image of a particular block level

    :param directory: The directory where the images are stored
    :param block level: [int] level 1: 1 histogram, level 2: 4 blocks --> 4 histograms concatenated, level 3: 16 blocks --> 16 histograms concatenated
    :param bins: number of bins of the histograms
    :param notQuery: If the image is a query image or not
    :return: A dictionary of histograms."""
    
    hist_dict = {}
    for filename in os.scandir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0]
            if not query:
                f_name = f_name.split('_')[1]

            image = cv2.imread(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab) 

            
            #spliting the image into blocks
            n_patches = int((2**blockLevel)/2)
            
            M = image.shape[0]//n_patches
            N = image.shape[1]//n_patches

            tiles = [image[x:x+M,y:y+N] for x in range(0,image.shape[0]-image.shape[0]%n_patches,M) for y in range(0,image.shape[1]-image.shape[1]%n_patches,N)]
            concat_hist_ch1 = []
            concat_hist_ch2 = []
            concat_hist_ch3 = []

            concat_hist_ch1 = np.float32(concat_hist_ch1)
            concat_hist_ch2 = np.float32(concat_hist_ch2)
            concat_hist_ch3 = np.float32(concat_hist_ch3)


            for tile in tiles:

                hist_ch1 = cv2.calcHist([tile], [0], None, [bins], [0, 255])
                hist_ch2 = cv2.calcHist([tile], [1], None, [bins], [0, 255])
                hist_ch3 = cv2.calcHist([tile], [2], None, [bins], [0, 255])

                hist_ch1 /= hist_ch1.sum()
                hist_ch2 /= hist_ch2.sum()
                hist_ch3 /= hist_ch3.sum()  
                
                concat_hist_ch1 = np.append(concat_hist_ch1,hist_ch1)
                concat_hist_ch2 = np.append(concat_hist_ch2,hist_ch2)
                concat_hist_ch3 = np.append(concat_hist_ch3,hist_ch3)
            
            hist_dict[f_name] = (histograms(concat_hist_ch1, concat_hist_ch2, concat_hist_ch3))

    return hist_dict




def get_block_histograms_multiLevel(directory, output_name, blockIniLevel,blockEndLevel, bins, query):
    """
    Calculate and concatenate histograms made from parts of the image in different block levels
    
    :param directory: The directory where the images are stored
    :param blockIniLevel: [int] 
    :param blockEndLevel: [int] 
                                level 1: 1 histogram, level 2: 4 blocks --> 4 histograms concatenated, level 3: 16 blocks --> 16 histograms concatenated
    :param bins: number of bins of the histograms
    :param notQuery: If the image is a query image or not
    :return: A dictionary of histograms.
    """
    hist_dict = {}
    for filename in os.scandir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0]
            if not query:
                f_name = f_name.split('_')[1]

            image = cv2.imread(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab) 

            #spliting the image into diferent level blocks

            concat_hist_ch1 = []
            concat_hist_ch2 = []
            concat_hist_ch3 = []

            concat_hist_ch1 = np.float32(concat_hist_ch1)
            concat_hist_ch2 = np.float32(concat_hist_ch2)
            concat_hist_ch3 = np.float32(concat_hist_ch3)

            for blockLevel in range (blockIniLevel,blockEndLevel+1):

                n_patches = int((2**blockLevel)/2)
                M = image.shape[0]//n_patches
                N = image.shape[1]//n_patches

                tiles = [image[x:x+M,y:y+N] for x in range(0,image.shape[0]-image.shape[0]%n_patches,M) for y in range(0,image.shape[1]-image.shape[1]%n_patches,N)]
            
                for index,tile in enumerate(tiles):

                    hist_ch1 = cv2.calcHist([tile], [0], None, [bins], [0, 255])
                    hist_ch2 = cv2.calcHist([tile], [1], None, [bins], [0, 255])
                    hist_ch3 = cv2.calcHist([tile], [2], None, [bins], [0, 255])

                    hist_ch1 /= hist_ch1.sum()
                    hist_ch2 /= hist_ch2.sum()
                    hist_ch3 /= hist_ch3.sum()  

                    concat_hist_ch1 = np.append(concat_hist_ch1,hist_ch1)
                    concat_hist_ch2 = np.append(concat_hist_ch2,hist_ch1)
                    concat_hist_ch3 = np.append(concat_hist_ch3,hist_ch3)
            
            hist_dict[f_name] = (histograms(concat_hist_ch1, concat_hist_ch2, concat_hist_ch3))

    return hist_dict


def get_histograms(directory, output_name, colorSpace, query, with_mask):
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
            f_name = filename.name.split('.')[0]
            if not query:
                f_name = f_name.split('_')[1]

            image = cv2.imread(f)

            if colorSpace == "HSV":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)           
            elif colorSpace == "LAB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)          
            elif colorSpace == "YCrCb":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            elif colorSpace == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if with_mask:    
                mask_name = os.path.join(directory, output_name + "/" + f_name + ".png")
                mask = cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE)
                if colorSpace == "HSV":
                    hist_ch1 = cv2.calcHist([image], [0], mask, [180], [0, 179])
                else:                
                    hist_ch1 = cv2.calcHist([image], [0], mask, [256], [0, 255])
                hist_ch2 = cv2.calcHist([image], [1], mask, [256], [0, 255])
                hist_ch3 = cv2.calcHist([image], [2], mask, [256], [0, 255])
            
            else:
                if colorSpace == "HSV":
                    hist_ch1 = cv2.calcHist([image], [0], None, [180], [0, 179])
                else:                
                    hist_ch1 = cv2.calcHist([image], [0], None, [256], [0, 255])
                hist_ch2 = cv2.calcHist([image], [1], None, [256], [0, 255])
                hist_ch3 = cv2.calcHist([image], [2], None, [256], [0, 255])

            hist_ch1 /= hist_ch1.sum()
            hist_ch2 /= hist_ch2.sum()
            hist_ch3 /= hist_ch3.sum()
            
            hist_dict[f_name] = (histograms(hist_ch1, hist_ch2, hist_ch3))
    return hist_dict
