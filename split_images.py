from main import *


# A function that takes two arguments: directory_query and directory_output. It then iterates through
# the directory_query and checks if the file is a jpg. If it is, it reads the image, erodes it, and
# then thresholds it. It then creates a mask_close and checks if there are two images or one image. If
# there are two images, it splits the image and saves the images in the directory_output. If there is
# only one image, it saves the image in the directory_output.
def split_images1(directory_query, directory_output):
    for filename in os.scandir(directory_query):
        f = os.path.join(directory_query, filename)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0]

            image = cv2.imread(f,0)
            kernel = np.ones((2,2),np.uint8)
            image_e= cv2.erode(image,kernel,iterations = 2)
            a,imgt = cv2.threshold(image_e, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            size = 5
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*size+1, 2*size+1))
            mask_close = cv2.morphologyEx(imgt, cv2.MORPH_CLOSE, element, iterations=5)
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
                    elif len(row_indexes) == 4:
                        two_images+=1
                        middle = (row_indexes[1] + row_indexes[2])/2
                        list_middles.append(middle)
                    elif len(row_indexes)>2 and len(row_indexes)<6:
                        two_images+=1
                        middle = (row_indexes[1] + row_indexes[2])/2
                        list_middles.append(middle)

            if two_images>one_image:
                real_middle = int(max(set(list_middles), key = list_middles.count))#int(np.mean(list_middles))

                first_img = mask_close[:,:real_middle]
                second_img = mask_close[:,real_middle:]

                #De moment ho guardo tot per poder veure que està fent, realment només hauria de guardar les màscares dividides
                cv2.imwrite(directory_output + f_name + '.jpg',image)
                cv2.imwrite(directory_output  + f_name + '_maskraw'+ '.jpg', mask_close)
                cv2.imwrite(directory_output + f_name + '_mask1' + '.jpg', first_img)
                cv2.imwrite(directory_output +  f_name + '_mask2' + '.jpg', second_img)
            else:
                cv2.imwrite(directory_output + f_name + '.jpg',image)
                cv2.imwrite(directory_output  + f_name + '_maskraw'+ '.jpg', mask_close)
    return

        
from main import *


# A function that takes two arguments: directory_query and directory_output. It then iterates through
# the directory_query and checks if the file is a jpg. If it is, it reads the image, erodes it, and
# then thresholds it. It then creates a mask_close and checks if there are two images or one image. If
# there are two images, it splits the image and saves the images in the directory_output. If there is
# only one image, it saves the image in the directory_output.
def split_images_v2(directory_query, directory_output):
    for filename in os.scandir(directory_query):
        f = os.path.join(directory_query, filename)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0]

            image = cv2.imread(f)
            kernel = np.ones((2,2),np.uint8)
            image_e= cv2.erode(image,kernel,iterations = 2)
            ret,imgt = cv2.threshold(image_e,90,255,cv2.THRESH_BINARY_INV)

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
                    elif len(row_indexes) == 4:
                        two_images+=1
                        middle = (row_indexes[1] + row_indexes[2])/2
                        list_middles.append(middle)
                    elif len(row_indexes)>2 and len(row_indexes)<6:
                        two_images+=1
                        middle = (row_indexes[1] + row_indexes[2])/2
                        list_middles.append(middle)

            if two_images>one_image:
                real_middle = int(max(set(list_middles), key = list_middles.count))#int(np.mean(list_middles))

                first_img = mask_close[:,:real_middle]
                second_img = mask_close[:,real_middle:]

                #De moment ho guardo tot per poder veure que està fent, realment només hauria de guardar les màscares dividides
                cv2.imwrite(directory_output + f_name + '.jpg',image)
                cv2.imwrite(directory_output  + f_name + '_maskraw'+ '.jpg', mask_close)
                cv2.imwrite(directory_output + f_name + '_mask1' + '.jpg', first_img)
                cv2.imwrite(directory_output +  f_name + '_mask2' + '.jpg', second_img)
            else:
                cv2.imwrite(directory_output + f_name + '.jpg',image)
                cv2.imwrite(directory_output  + f_name + '_maskraw'+ '.jpg', mask_close)
    return

def split_images3(directory_query, directory_output):
    for filename in os.scandir(directory_query):
        f = os.path.join(directory_query, filename)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0]

            image = cv2.imread(f,0)
            #kernel = np.ones((2,2),np.uint8)
            #image_e= cv2.erode(image,kernel,iterations = 15)
            size = 5
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*size+1, 2*size+1))
            image_o = cv2.morphologyEx(image, cv2.MORPH_OPEN, element, iterations=5)
            a,imgt = cv2.threshold(image_o, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            size = 5
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*size+1, 2*size+1))
            mask_close = cv2.morphologyEx(imgt, cv2.MORPH_CLOSE, element, iterations=5)
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
                    elif len(row_indexes) == 4:
                        two_images+=1
                        middle = (row_indexes[1] + row_indexes[2])/2
                        list_middles.append(middle)
                    elif len(row_indexes)>2 and len(row_indexes)<6:
                        two_images+=1
                        middle = (row_indexes[1] + row_indexes[2])/2
                        list_middles.append(middle)

            if two_images>one_image:
                real_middle = int(max(set(list_middles), key = list_middles.count))#int(np.mean(list_middles))

                first_img = mask_close[:,:real_middle]
                second_img = mask_close[:,real_middle:]

                #De moment ho guardo tot per poder veure que està fent, realment només hauria de guardar les màscares dividides
                cv2.imwrite(directory_output + f_name + '.jpg',image)
                cv2.imwrite(directory_output  + f_name + '_maskraw'+ '.jpg', mask_close)
                cv2.imwrite(directory_output + f_name + '_mask1' + '.jpg', first_img)
                cv2.imwrite(directory_output +  f_name + '_mask2' + '.jpg', second_img)
            else:
                cv2.imwrite(directory_output + f_name + '.jpg',image)
                cv2.imwrite(directory_output  + f_name + '_maskraw'+ '.jpg', mask_close)
    return


base_dir = '../'
name_query='qsd2_w2/'
directory_query = base_dir + name_query
directory_output = base_dir + 'predictions/'
# split_images3(directory_query, directory_output)

