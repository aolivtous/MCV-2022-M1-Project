from numpy import int8
from main import *

class histograms: 
    def __init__(self, hist_gray, hist_ch1, hist_ch2, hist_ch3): 
        self.hist_gray = hist_gray
        self.hist_ch1 = hist_ch1
        self.hist_ch2 = hist_ch2
        self.hist_ch3 = hist_ch3

def get_block_histograms(image, n_patches, bins, has_boundingbox, is_query, text_mask, descriptor):
    
    """Calculate and concatenate histograms made from parts of the image of a particular block level

    :param image: image you want to get the histogram descriptors
    :param n_patches: size of the division grid --> n*n  
    :param bins: number of bins of the histograms
    :param is_query: If the image is a query image or not
    :text_mask: binary mask that contains the text box
    :return: A dictionary of histograms."""

    if descriptor == 'color':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab) 

    elif descriptor == 'texture':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = feature.local_binary_pattern(image, 8, 2, method="uniform").astype(np.uint8)
    
    if(is_query and has_boundingbox): 
        th, text_mask = cv2.threshold(text_mask, 128, 255, cv2.THRESH_BINARY)      
    
    n_patches = int(n_patches)
    
    M = image.shape[0]//n_patches
    N = image.shape[1]//n_patches

    tiles = [image[x:x+M,y:y+N] for x in range(0,image.shape[0]-image.shape[0]%n_patches,M) for y in range(0,image.shape[1]-image.shape[1]%n_patches,N)]
    
    tiles_mask = []
    if(is_query and has_boundingbox):
        tiles_mask = [text_mask[x:x+M,y:y+N] for x in range(0,text_mask.shape[0]-text_mask.shape[0]%n_patches,M) for y in range(0,text_mask.shape[1]-text_mask.shape[1]%n_patches,N)]
    
    concat_hist_gray = np.float32(np.array([]))
    concat_hist_ch1 = np.float32(np.array([]))
    concat_hist_ch2 = np.float32(np.array([]))
    concat_hist_ch3 = np.float32(np.array([]))

    for idx, tile in enumerate(tiles):
        
        if descriptor == 'color':
            if(is_query and has_boundingbox):
                hist_ch1 = cv2.calcHist([tile], [0], tiles_mask[idx], [bins], [0, 255])
                hist_ch2 = cv2.calcHist([tile], [1], tiles_mask[idx], [bins], [0, 255])
                hist_ch3 = cv2.calcHist([tile], [2], tiles_mask[idx], [bins], [0, 255])
            else:
                hist_ch1 = cv2.calcHist([tile], [0], None, [bins], [0, 255])
                hist_ch2 = cv2.calcHist([tile], [1], None, [bins], [0, 255])
                hist_ch3 = cv2.calcHist([tile], [2], None, [bins], [0, 255])

            # We know we are producing some NaNs with this operation, we clean them later
            with np.errstate(divide='ignore',invalid='ignore'):
                hist_ch1 /= hist_ch1.sum()
                hist_ch2 /= hist_ch2.sum()
                hist_ch3 /= hist_ch3.sum()
                
            concat_hist_gray = None
            concat_hist_ch1 = np.append(concat_hist_ch1,hist_ch1)
            concat_hist_ch2 = np.append(concat_hist_ch2,hist_ch2)
            concat_hist_ch3 = np.append(concat_hist_ch3,hist_ch3)
        
        elif descriptor=='texture':
            if(is_query and has_boundingbox):
                hist_gray = cv2.calcHist([tile], [0], tiles_mask[idx], [bins], [0, 255])
            else:
                hist_gray = cv2.calcHist([tile], [0], None, [bins], [0, 255])

            with np.errstate(divide='ignore',invalid='ignore'):
                hist_gray/=hist_gray.sum()

            concat_hist_gray = np.append(concat_hist_gray,hist_gray)
            concat_hist_ch1 = None
            concat_hist_ch2 = None
            concat_hist_ch3 = None


    
    return (histograms(concat_hist_gray, concat_hist_ch1, concat_hist_ch2, concat_hist_ch3))


