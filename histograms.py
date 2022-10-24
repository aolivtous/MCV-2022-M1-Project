from numpy import hamming, int8
from main import *

class histograms: 
    def __init__(self, hist_lbp, coeffs_dct, hist_ch1, hist_ch2, hist_ch3): 
        self.hist_lbp = hist_lbp
        self.coeffs_dct = coeffs_dct
        self.hist_ch1 = hist_ch1
        self.hist_ch2 = hist_ch2
        self.hist_ch3 = hist_ch3

def get_block_histograms(image, n_patches, bins, has_boundingbox, is_query, text_mask, descriptors):
    
    """Calculate and concatenate histograms made from parts of the image of a particular block level

    :param image: image you want to get the histogram descriptors
    :param n_patches: size of the division grid --> n*n  
    :param bins: number of bins of the histograms
    :param is_query: If the image is a query image or not
    :text_mask: binary mask that contains the text box
    :return: A dictionary of histograms."""

    for i in range(len(descriptors)):

        n_patches = int(n_patches)
        
        M = image.shape[0]//n_patches
        N = image.shape[1]//n_patches

        if 'color' in descriptors:
            image_color = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            tiles_color = [image_color[x:x+M,y:y+N] for x in range(0,image_color.shape[0]-image_color.shape[0]%n_patches,M) for y in range(0,image_color.shape[1]-image_color.shape[1]%n_patches,N)]

        if 'texture' in descriptors:
            image_texture = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            tiles_texture = [image_texture[x:x+M,y:y+N] for x in range(0,image_texture.shape[0]-image_texture.shape[0]%n_patches,M) for y in range(0,image_texture.shape[1]-image_texture.shape[1]%n_patches,N)]  
           
        if(is_query and has_boundingbox): 
            th, text_mask = cv2.threshold(text_mask, 128, 255, cv2.THRESH_BINARY)      


        #tiles = [image[x:x+M,y:y+N] for x in range(0,image.shape[0]-image.shape[0]%n_patches,M) for y in range(0,image.shape[1]-image.shape[1]%n_patches,N)]
        
        tiles_mask = []
        if(is_query and has_boundingbox):
            tiles_mask = [text_mask[x:x+M,y:y+N] for x in range(0,text_mask.shape[0]-text_mask.shape[0]%n_patches,M) for y in range(0,text_mask.shape[1]-text_mask.shape[1]%n_patches,N)]
        
        concat_hist_lbp = np.float32(np.array([]))
        concat_coeffs_dct = np.float32(np.array([]))
        concat_hist_ch1 = np.float32(np.array([]))
        concat_hist_ch2 = np.float32(np.array([]))
        concat_hist_ch3 = np.float32(np.array([]))

        for idx in range((n_patches**2)-1):
            
            if 'color' in descriptors:
                if(is_query and has_boundingbox):
                    hist_ch1 = cv2.calcHist([tiles_color[idx]], [0], tiles_mask[idx], [bins], [0, 255])
                    hist_ch2 = cv2.calcHist([tiles_color[idx]], [1], tiles_mask[idx], [bins], [0, 255])
                    hist_ch3 = cv2.calcHist([tiles_color[idx]], [2], tiles_mask[idx], [bins], [0, 255])
                else:
                    hist_ch1 = cv2.calcHist([tiles_color[idx]], [0], None, [bins], [0, 255])
                    hist_ch2 = cv2.calcHist([tiles_color[idx]], [1], None, [bins], [0, 255])
                    hist_ch3 = cv2.calcHist([tiles_color[idx]], [2], None, [bins], [0, 255])

                # We know we are producing some NaNs with this operation, we clean them later
                with np.errstate(divide='ignore',invalid='ignore'):
                    hist_ch1 /= hist_ch1.sum()
                    hist_ch2 /= hist_ch2.sum()
                    hist_ch3 /= hist_ch3.sum()
   
                if 'texture' not in descriptors:
                    concat_hist_gray = None
                concat_hist_ch1 = np.append(concat_hist_ch1,hist_ch1)
                concat_hist_ch2 = np.append(concat_hist_ch2,hist_ch2)
                concat_hist_ch3 = np.append(concat_hist_ch3,hist_ch3)
            
            if 'texture' in descriptors:
                #lbp
                if(is_query and has_boundingbox):
                    hist_lbp = cv2.calcHist([tiles_texture[idx]], [0], tiles_mask[idx], [bins], [0, 255])
                else:
                    hist_lbp = cv2.calcHist([tiles_texture[idx]], [0], None, [bins], [0, 255])

                with np.errstate(divide='ignore',invalid='ignore'):
                    hist_lbp/=hist_lbp.sum()

                concat_hist_lbp = np.append(concat_hist_lbp,hist_lbp)

                
                
                #dct
                patch_texture = tiles_texture[idx]
                m,n=patch_texture.shape
                if (m % 2) != 0:
                    patch_texture = np.append(patch_texture, [np.zeros(n)], axis=0)
                m,n=patch_texture.shape
                if (n % 2) != 0:
                    patch_texture = np.append(patch_texture, np.zeros((m,1)), axis=1)
                

                patch_float = np.float64(patch_texture)/255.0  # float conversion/scale
                patch_texture_dct_rang1 = cv2.dct(patch_float)
                patch_texture_dct = np.uint8(patch_texture_dct_rang1*255.0) 

                N=100 #number of coefficients to consider
                zigzag_vector = np.concatenate([np.diagonal(patch_texture_dct[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-patch_texture_dct.shape[0], patch_texture_dct.shape[0])])

                concat_coeffs_dct = np.append(concat_coeffs_dct,zigzag_vector[:N])

                if 'color' not in descriptors:
                    concat_hist_ch1 = None
                    concat_hist_ch2 = None
                    concat_hist_ch3 = None


    
    return (histograms(concat_hist_lbp, concat_coeffs_dct, concat_hist_ch1, concat_hist_ch2, concat_hist_ch3))