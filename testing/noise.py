import global_variables
import cv2
import numpy as np
import os
import math

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

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