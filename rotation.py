import global_variables
import cv2
import numpy as np

def rotation_check(image, f_name):
    # Applying hough to detect lines
    image_cpy = image.copy()
    edges = cv2.Canny(image, 50, 150, apertureSize = 3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    # Represent the lines over the image
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_cpy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    # Plot image copy
    cv2.imshow('image_cpy', image_cpy)
    cv2.waitKey(0)

    # Get the angle of the lines
    angles = []
    for line in lines:
        rho, theta = line[0]
        # Get the number of times the angle is repeated only if its lower than 45 degrees
        if theta < np.pi / 4 or (theta > 3 * np.pi / 4 and theta < 5 * np.pi / 4) or theta > 7 * np.pi / 4:
            angles.append(theta)
    
    # Get length of the lines
    lengths = []
    angles = []
    for line in lines:
        rho, theta = line[0]
        if theta < np.pi / 4 or (theta > 3 * np.pi / 4 and theta < 5 * np.pi / 4) or theta > 7 * np.pi / 4:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            lengths.append(np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2))
            angles.append(theta)


    # Get the longest line and its angle

    print(lengths)
    print(angles)
    # Check if angles is empty
    if not angles:
        return image
    
    # Get the most repeated angle
    angle = max(set(angles), key = angles.count)
    # Get the angle in degrees
    if angle < np.pi / 4:
        angle_deg = angle * 180 / np.pi
    else:
        angle_deg = (angle - np.pi) * 180 / np.pi
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle_deg, 1)
    # Rotate the image
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=get_avg_corners_color(image))
    # Save the rotated image
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_rotated.png', rotated_image)
    # Print the angle in degrees in a file
    with open(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_angle.txt', 'w') as f:
        f.write(str(angle_deg))
    return rotated_image, M
        
def get_avg_corners_color(image):
    # Get average color of the image corners
    # top left
    avg_color_per_row = np.average(image[0:10, 0:10], axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color_tl = avg_color
    # top right
    avg_color_per_row = np.average(image[0:10, image.shape[1]-10:image.shape[1]], axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color_tr = avg_color
    # bottom left
    avg_color_per_row = np.average(image[image.shape[0]-10:image.shape[0], 0:10], axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color_bl = avg_color
    # bottom right
    avg_color_per_row = np.average(image[image.shape[0]-10:image.shape[0], image.shape[1]-10:image.shape[1]], axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color_br = avg_color
    # Get the average color of the corners
    avg_color_corners = (avg_color_tl + avg_color_tr + avg_color_bl + avg_color_br) / 4
    return avg_color_corners