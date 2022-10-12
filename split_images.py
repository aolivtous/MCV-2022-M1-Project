from main import *
import statistics

base_dir = "../"
name_query = 'qsd2_w2/'
directory_query = base_dir + name_query
query = True
directory_output = base_dir + 'output/'

for filename in os.scandir(directory_query):
    f = os.path.join(directory_query, filename)
    # checking if it is a file
    if f.endswith('.jpg'):
        f_name = filename.name.split('.')[0]
        if not query:
            f_name = f_name.split('_')[1]

        image = cv2.imread(f,0)
        ret,imgt = cv2.threshold(image,100,255,cv2.THRESH_BINARY_INV)
        # show the image, provide window name first
        #cv2.imshow('',imgt)
        # add wait key. window waits until user presses a key
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        size = 5
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*size+1, 2*size+1))
        mask_close = cv2.morphologyEx(imgt, cv2.MORPH_CLOSE, element, iterations=10)
        M,N = mask_close.shape
        one_image = 0
        two_images = 0
        list_middles=[]
        for i in range(M):
            if sum(mask_close[i])>100:
                row_indexes =[]
                for j in range(N-1):
                    if mask_close[i][j] != mask_close[i][j+1]:
                        row_indexes.append(j)
                if len(row_indexes) == 2:
                    one_image+=1
                if len(row_indexes)>2 and len(row_indexes)<6:
                    two_images+=1
                    middle = (row_indexes[1] + row_indexes[2])/2
                    list_middles.append(middle)

        if two_images>one_image:
            real_middle = int(np.mean(list_middles))

            first_img = mask_close[:,:real_middle]
            second_img = mask_close[:,real_middle:]


            cv2.imwrite(directory_output + f_name + '.jpg',image)
            cv2.imwrite(directory_output  + f_name + '_maskraw'+ '.jpg', mask_close)
            cv2.imwrite(directory_output + f_name + '_mask1' + '.jpg', first_img)
            cv2.imwrite(directory_output +  f_name + '_mask2' + '.jpg', second_img)
        else:
            cv2.imwrite(directory_output + f_name + '.jpg',image)
            cv2.imwrite(directory_output  + f_name + '_maskraw'+ '.jpg', mask_close)


        

'''
        # do the 2D fourier transform
        fft_img = np.fft.fft2(image)

        M,N = fft_img.shape

        #for i in range(len(image_freq)):
         #   for j in range(len(image_freq)):
          #     if i>300 and i<M-300 and j>300 and j<N-300:
           #         image_freq[i][j] = 255
        
        # shift FFT to the center
        fft_img_shift = np.fft.fftshift(fft_img)

        # extract real and phases
        real = fft_img_shift.real
        phases = fft_img_shift.imag

        # modify real part, put your modification here
        def imgRadius(img, radius):
            result = np.zeros(img.shape,np.float64)
            centerX = (img.shape[0])/2
            centerY = (img.shape[1])/2
            for m in range(img.shape[0]):
                for n in range(img.shape[1]):
                    if math.sqrt((m-centerX)**2+(n-centerY)**2) < radius:
                        result[m,n] = img[m,n]
            return result

            
        real_mod = imgRadius(real,40)

        real_mod_amp = np.log(np.abs(real_mod))

        # create an empty complex array with the shape of the input image
        fft_img_shift_mod = np.empty(real.shape, dtype=complex)

        # insert real and phases to the new file
        fft_img_shift_mod.real = real_mod
        fft_img_shift_mod.imag = phases

        # reverse shift
        fft_img_mod = np.fft.ifftshift(fft_img_shift_mod)

        # reverse the 2D fourier transform
        img_mod = np.fft.ifft2(fft_img_mod)

        # using np.abs gives the scalar value of the complex number
        # with img_mod.real gives only real part. Not sure which is proper
        img_mod = np.abs(img_mod)


        plt.imshow(img_mod, cmap='gray')
        plt.show()
'''

