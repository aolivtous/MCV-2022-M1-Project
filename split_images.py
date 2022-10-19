import global_variables
import cv2
import os

# A function that takes two arguments: dir_query and directory_output. It then iterates through
# the dir_query and checks if the file is a jpg. If it is, it reads the image, erodes it, and
# then thresholds it. It then creates a mask_close and checks if there are two images or one image. If
# there are two images, it splits the image and saves the images in the directory_output. If there is
# only one image, it saves the image in the directory_output.
def split_images(image, f_name):

    m = os.path.join(f'{global_variables.dir_results}{f_name}.png')
    mask = cv2.imread(m,0)

    height, width, channels = image.shape
    one_image = 0
    two_images = 0
    list_middles=[]
    for i in range(height):
        if sum(mask[i])>100: #width*255*0.5:
            row_indexes =[]
            for j in range(width-2):
                if mask[i][j] != mask[i][j+1]:
                    row_indexes.append(j)
            if len(row_indexes) == 2:
                one_image+=1
            elif len(row_indexes) == 4:
                two_images+=1
                middle = (row_indexes[1] + row_indexes[2])/2
                list_middles.append(middle)
            # elif len(row_indexes)>2 and len(row_indexes)<6:
            #     two_images+=1
            #     middle = (row_indexes[1] + row_indexes[2])/2
            #     list_middles.append(middle)

    if two_images > one_image:
        real_middle = int(max(set(list_middles), key = list_middles.count))#int(np.mean(list_middles))

        first_img = image[:,:real_middle]
        second_img = image[:,real_middle:]

        first_mask = mask[:,:real_middle]
        second_mask = mask[:,real_middle:]
    
        cv2.imwrite(f'{global_variables.dir_query}{f_name}_part1.jpg', first_img)
        cv2.imwrite(f'{global_variables.dir_query}{f_name}_part2.jpg', second_img)
        cv2.imwrite(f'{global_variables.dir_query_aux}{f_name}_part1_mask.png', first_mask)
        cv2.imwrite(f'{global_variables.dir_query_aux}{f_name}_part2_mask.png', second_mask)
    return