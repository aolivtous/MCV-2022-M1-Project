from numpy import hamming, int8
from main import *
import global_variables

class histograms: 
    def __init__(self, feature, coeffs_dct, hist_ch1, hist_ch2, hist_ch3): 
        #self.hist_lbp = hist_lbp
        self.feature = feature
        self.coeffs_dct = coeffs_dct
        self.hist_ch1 = hist_ch1
        self.hist_ch2 = hist_ch2
        self.hist_ch3 = hist_ch3
        #self.concatenation = concatenation

def get_block_histograms(image, has_boundingbox, is_query, text_mask):
    
    """Calculate and concatenate histograms made from parts of the image of a particular block level

    :param image: image you want to get the histogram descriptors
    :param n_patches: size of the division grid --> n*n  
    :param bins: number of bins of the histograms
    :param is_query: If the image is a query image or not
    :text_mask: binary mask that contains the text box
    :return: A dictionary of histograms."""
    weights = global_variables.weights
    n_patches = int(global_variables.n_patches)
    n_bins = int(global_variables.n_bins)
    
    M = image.shape[0]//n_patches
    N = image.shape[1]//n_patches

    tiles_color = tiles_texture = np.array([])
    if weights['color']:
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        tiles_color = [image_color[x:x+M,y:y+N] for x in range(0,image_color.shape[0]-image_color.shape[0]%n_patches,M) for y in range(0,image_color.shape[1]-image_color.shape[1]%n_patches,N)]

    if weights['texture']:
        image_texture = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tiles_texture = [image_texture[x:x+M,y:y+N] for x in range(0,image_texture.shape[0]-image_texture.shape[0]%n_patches,M) for y in range(0,image_texture.shape[1]-image_texture.shape[1]%n_patches,N)]  

    tiles_mask = []
    if(is_query and has_boundingbox): 
        _, text_mask = cv2.threshold(text_mask, 128, 255, cv2.THRESH_BINARY)      
        tiles_mask = [text_mask[x:x+M,y:y+N] for x in range(0,text_mask.shape[0]-text_mask.shape[0]%n_patches,M) for y in range(0,text_mask.shape[1]-text_mask.shape[1]%n_patches,N)]
    
    concat_feature = np.float32(np.array([]))
    concat_coeffs_dct = np.float32(np.array([]))
    concat_coeffs_dct_norm = np.float32(np.array([]))
    concat_hist_ch1 = np.float32(np.array([]))
    concat_hist_ch2 = np.float32(np.array([]))
    concat_hist_ch3 = np.float32(np.array([]))
    #concat_features = np.float32(np.array([]))

    first_time = True


    # sift.detectAndCompute() function finds the keypoints and the descriptors of an image. You can pass a mask if you want to search only a part of image. 
    # Each keypoint is a special structure which has many attributes 
    # like its (x,y) coordinates, size of the meaningful neighbourhood, 
    # angle which specifies its orientation, response that specifies strength of keypoints etc.
    #concat_feature = ()

    if weights['feature']:  
        image_feature = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if global_variables.methods_search['default']['feature_algorithm'] == 'SIFT':
            sift = cv2.SIFT_create()
            if (is_query and has_boundingbox):
                kp, descrip = sift.detectAndCompute(image_feature, text_mask)
            else:
                kp, descrip = sift.detectAndCompute(image_feature, None)
        elif global_variables.methods_search['default']['feature_algorithm'] == 'SURF':
            # Here we set the Hessian threshold to 400
            hess_thr=400
            surf = cv2.SURF_create(hess_thr)
            if (is_query and has_boundingbox): 
                kp, descrip = surf.detectAndCompute(image_feature, text_mask)
            else:
                kp, descrip = surf.detectAndCompute(image_feature, None)
        else:
            print('Feature method not found')
            exit(1)

        
        # Cal fer aquest pas per poder guardar en el pickle els resultats de la db
        # concat_feature = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, descrip)
        concat_feature = descrip

        

    for idx in range((n_patches**2)):
        
        # Not considering the borders of the (cropped) image
        # if idx < n_patches or idx % n_patches == 0 or idx % n_patches == n_patches-1 or idx >= (n_patches**2)-n_patches:
        #     continue
        
        if weights['color']:
            if(is_query and has_boundingbox):
                hist_ch1 = cv2.calcHist([tiles_color[idx]], [0], tiles_mask[idx], [n_bins], [0, 255])
                hist_ch2 = cv2.calcHist([tiles_color[idx]], [1], tiles_mask[idx], [n_bins], [0, 255])
                hist_ch3 = cv2.calcHist([tiles_color[idx]], [2], tiles_mask[idx], [n_bins], [0, 255])
            else:
                hist_ch1 = cv2.calcHist([tiles_color[idx]], [0], None, [n_bins], [0, 255])
                hist_ch2 = cv2.calcHist([tiles_color[idx]], [1], None, [n_bins], [0, 255])
                hist_ch3 = cv2.calcHist([tiles_color[idx]], [2], None, [n_bins], [0, 255])

            # We know we are producing some NaNs with this operation, we clean them later
            with np.errstate(divide='ignore',invalid='ignore'):
                hist_ch1 /= hist_ch1.sum()
                hist_ch2 /= hist_ch2.sum()
                hist_ch3 /= hist_ch3.sum()

            concat_hist_ch1 = np.append(concat_hist_ch1,hist_ch1)
            concat_hist_ch2 = np.append(concat_hist_ch2,hist_ch2)
            concat_hist_ch3 = np.append(concat_hist_ch3,hist_ch3)
            #concatenation_features = np.append(concatenation_features, hist_ch1)
            #concatenation_features = np.append(concatenation_features, hist_ch2)
            #concatenation_features = np.append(concatenation_features, hist_ch3)
        
        if weights['texture']:
            """#lbp
            if(is_query and has_boundingbox):
                tiles_texture_lbp = feature.local_binary_pattern(tiles_texture[idx],8,2,method='uniform').astype(np.uint8)
                hist_lbp = cv2.calcHist([tiles_texture_lbp], [0], tiles_mask[idx], [bins], [0, 255])
            else:
                tiles_texture_lbp = feature.local_binary_pattern(tiles_texture[idx],8,2,method='uniform').astype(np.uint8)
                hist_lbp = cv2.calcHist([tiles_texture_lbp], [0], None, [bins], [0, 255])

            with np.errstate(divide='ignore',invalid='ignore'):
                hist_lbp/=hist_lbp.sum()

            concat_hist_lbp = np.append(concat_hist_lbp,hist_lbp)"""
            
            #dct
            # If all pixels at tiles_mask (array of arrays) are 0 (black), then we append to concat_coeffs_dct a vector of Nans
            patch_texture = tiles_texture[idx]
            X = 400 # Number of coefficients to consider

            if(is_query and has_boundingbox):
                # Get a flattened 1D view of 2D numpy array
                flatten_tiles_mask = np.ravel(tiles_mask[idx])
                # Check if all value in 2D array are zero
                # Percentage of zero values in flatten_tiles_mask
                percentage_mask = np.count_nonzero(flatten_tiles_mask == 0) / flatten_tiles_mask.size
                # if np.all(flatten_tiles_mask == 0):
                if percentage_mask > 0.5:
                    if(first_time):
                        first_time = False
                    # Append a vector of X Nans to concat_coeffs_dct and continue to next iteration
                    concat_coeffs_dct = np.append(concat_coeffs_dct, np.full(X, np.nan))
                    continue

            m,n=patch_texture.shape
            if (m % 2) != 0:
                patch_texture = np.append(patch_texture, [np.zeros(n)], axis=0)
            m,n=patch_texture.shape
            if (n % 2) != 0:
                patch_texture = np.append(patch_texture, np.zeros((m,1)), axis=1)

            patch_float = np.float64(patch_texture)/255.0  
            patch_texture_dct = cv2.dct(patch_float)
            
            zigzag_vector = np.concatenate([np.diagonal(patch_texture_dct[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-patch_texture_dct.shape[0], patch_texture_dct.shape[0])])

            concat_coeffs_dct = np.append(concat_coeffs_dct, zigzag_vector[:X])

            #norm_hist_image_coeffs_dct = (concat_coeffs_dct - concat_coeffs_dct.mean()) / concat_coeffs_dct.std()
            #concatenation_features = np.append(concatenation_features, norm_hist_image_coeffs_dct)

    # if weights['texture']:
    #     concat_coeffs_dct_norm = (concat_coeffs_dct - concat_coeffs_dct.min()) / (concat_coeffs_dct.max() - concat_coeffs_dct.min())
        #concat_hist_lbp = (concat_hist_lbp - concat_hist_lbp.min()) / (concat_hist_lbp.max() - concat_hist_lbp.min())

                
    return (histograms(concat_feature, concat_coeffs_dct, concat_hist_ch1, concat_hist_ch2, concat_hist_ch3))
    #return (histograms(concat_hist_lbp, concat_coeffs_dct_norm, concat_hist_ch1, concat_hist_ch2, concat_hist_ch3))




