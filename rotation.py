import global_variables
import cv2
import numpy as np

def rotation_check(image, f_name):
    # Applying hough to detect lines
    image_cpy = image.copy()
    height, width = image.shape[:2]

    # edges = auto_canny(image)
    # Apply a tight cannny to detect edges
    edges = cv2.Canny(image, 200, 250)

    lines = cv2.HoughLinesP(edges, rho = 1, theta = 1*np.pi/180, threshold = 100, minLineLength = 100, maxLineGap = width)

    # Save result of canny
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_canny.png', edges)

    # If there are no lines detected, return the original image
    if lines is None:
        return image, False, 0, None
    
    # Get length of the lines
    final_length, final_angle, final_line = None, None, None
    min_height = height
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if  (y1 < height / 3 or y2 < height / 3) or (y1 > 2 * height / 3 or y2 > 2 * height / 3):
            # Get angle on radians
            angle = np.arctan2(y2 - y1, x2 - x1)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            mean_height = (y1 + y2) / 2
            
            # Plot lines in different colors for each part (range from -pi to pi)
            if (angle <= np.pi / 4 and angle >= - np.pi / 4) or (angle >= 3 * np.pi / 4 or angle <= - 3 * np.pi / 4):
                color = (0, 255, 0)
                if final_angle == None or mean_height < min_height:# length > final_length:
                    final_angle = angle
                    final_length = length
                    min_height = mean_height
                    final_line = [(x1, y1), (x2, y2)]
            else:
                color = (0, 0, 255)
            cv2.line(image_cpy, (x1, y1), (x2, y2), color, 2)

    if final_line:
        cv2.line(image_cpy, final_line[0], final_line[1], (255, 0, 0), 5)

    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_lines.png', image_cpy)

    # Check if angles is empty
    if not final_angle:
        return image, False, 0, None
    
    # Get the angle in degrees considering range pi to -pi
    angle_deg = final_angle * 180 / np.pi

    print(angle_deg)
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle_deg, 1)
    # Rotate the image

    # Transform the image to BGRA to avoid black borders
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    # rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=get_avg_corners_color(image))
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=cv2.BORDER_TRANSPARENT)
    # Save the rotated image
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_rotated.png', rotated_image)
    # Create a mask for the transparent pixels
    rotatation_mask = rotated_image[:,:,3] == 0
    # Format the mask to be binary
    rotatation_mask = rotatation_mask.astype(np.uint8) * 255
    cv2.imwrite(global_variables.dir_query + global_variables.dir_query_aux + f_name + '_rotated_mask.png', rotatation_mask)
    
    # Transform the image back to BGR
    rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGRA2BGR)

    return rotated_image, M, final_angle, rotatation_mask

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
        
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
