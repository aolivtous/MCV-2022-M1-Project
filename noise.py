import global_variables
import cv2
import numpy as np
import os
import math


dir_input = '../qsd1_w3/'
dir_input_non_augmented ='../qsd1_w3/non_augmented/'
#dir_output = '../denoised/'

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


# We find that SSIM is the best predictor to know if image needs to be denoised or not. Using median filter to all images of the query1 it has been found that
# for those pairs (image,median filtered) which obtain SSIM value less than 0.6 means that image has to be denoised, and for those pair which obtain SSIM
# value greater than 0.75 means that image doesn't need to be denoised
def noise_thresholds_def(dir_input):
    
    for filename in os.scandir(dir_input):
        f = os.path.join(dir_input, filename.name)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0]
            #print(f)
            image=cv2.imread(f)
            median = cv2.medianBlur(image,3)

            psnr = calculate_psnr(image,median)
            ssim = calculate_ssim(image,median)

            print(f'{f_name}/{ssim}')

            compare = np.concatenate((image,median),axis=1)

            # cv2.imshow(f'{f_name}', compare)
            # cv2.waitKey()
            # cv2.destroyAllWindows

    return

#noise_thresholds_def(dir_input)


def noise_thresholds_def2(dir_input, dir_input_non_augmented):
    
    for filename in os.scandir(dir_input):
        f = os.path.join(dir_input, filename.name)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0]
            #print(f)
            image=cv2.imread(f)
            original = cv2.imread(f'{dir_input_non_augmented}{filename.name}')

            psnr = calculate_psnr(image,original)
            ssim = calculate_ssim(image,original)

            print(f'{f_name}/ssim={ssim} psnr={psnr}')

            compare = np.concatenate((image,original),axis=1)

            # cv2.imshow(f'{f_name}', compare)
            # cv2.waitKey()
            # cv2.destroyAllWindows

    return

# noise_thresholds_def2(dir_input,dir_input_non_augmented)

def noise_ckeck_removal(image,f_name):

    median = cv2.medianBlur(image,3)
    ssim = calculate_ssim(image,median)
    to_be_denoised = False
    image_denoised = image
    if(ssim < 0.65):
        to_be_denoised = True
        image_denoised = median
    
    return to_be_denoised, image_denoised
















