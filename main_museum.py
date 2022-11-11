import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

# Internal modules
import global_variables
import utils
global_variables.init('name_query')

check_dirs = [
    global_variables.dir_museum, 
    global_variables.dir_museum + 'dominant_colors/', 
    global_variables.dir_museum + 'dcts/',
    global_variables.dir_museum + 'cluster_1/',
    global_variables.dir_museum + 'cluster_2/',
    global_variables.dir_museum + 'cluster_3/',
    global_variables.dir_museum + 'cluster_4/',
    global_variables.dir_museum + 'cluster_5/',
]

for dir in check_dirs:
    try:
        os.makedirs(dir)
    except FileExistsError:
        # Directory already exists
        pass

def main_museum():

    # If there are not enough arguments, exit the program.
    try:
        recalc_dominant_colors = bool(utils.str_to_bool(sys.argv[1]))
        recalc_dct = bool(utils.str_to_bool(sys.argv[2]))
    except:
        print(f'Exiting. Not enough arguments ({len(sys.argv) - 1} of 2)')
        exit(1)

    dominant_colors = {} 
    # { 
    #   DB_idx: {
    #       colors: Array of 3-Array BGR Format, 
    #       percent: Array of dominance of the color
    #   } 
    # }
    dct = {}
    # {
    #   DB_idx: Array of DCT Coefficients
    # }

    # Data Gathering
    if recalc_dominant_colors or recalc_dct:
        print('Data Gathering')
        for filename in tqdm(os.scandir(global_variables.dir_db)):
            f = os.path.join(global_variables.dir_db, filename)
            # checking if it is a file
            if f.endswith('.jpg'):
                f_name = filename.name.split('.')[0].split('_')[1]
                image = cv2.imread(f)
                print('Processing image: ', f_name)
                if recalc_dominant_colors:
                    chart = get_dominant_color(image, f_name, dominant_colors)
                    cv2.imwrite(global_variables.dir_museum + 'dominant_colors/' + f_name + '_dominant_color.jpg', chart)
                if recalc_dct:
                    calculate_dct(image, f_name, dct)
    
    # Save dominant colors in a pickle file
    if recalc_dominant_colors:
        try:
            with open(f'{global_variables.dir_museum}precomputed_dominant_colors.pkl', "wb" ) as f:
                pickle.dump(dominant_colors, f)
                print('Saved dominant colors pickle', dominant_colors)
        except:
            print('Error when trying to save the pickle file')
            print('Dominant colors pickle:', dominant_colors)
            exit(1)
    else:
        try:
            with open(f'{global_variables.dir_museum}precomputed_dominant_colors.pkl', "rb" ) as f:
                dominant_colors = pickle.load(f)
                # print('Read dominant colors pickle', dominant_colors)
        except:
            print('Exiting. No precomputed pickles found')
            exit(1)
    
    # Save dct in a pickle file
    if recalc_dct:
        try:
            with open(f'{global_variables.dir_museum}precomputed_dct.pkl', "wb" ) as f:
                pickle.dump(dct, f)
                print('Saved dct pickle', dct)
        except:
            print('Error when trying to save the pickle file')
            print('DCT pickle:', dct)
            exit(1)
    else:
        try:
            with open(f'{global_variables.dir_museum}precomputed_dct.pkl', "rb" ) as f:
                dct = pickle.load(f)
                # print('Read dct pickle', dct)
        except:
            print('Exiting. No precomputed pickles found')
            exit(1)

    # Join the two dictionaries and format dct
    data = {}
    for key in dominant_colors:
        data[key] = {
            'colors': dominant_colors[key]['colors'],
            'percent': dominant_colors[key]['percent'],
            'dct': dct[key]#[0] # Pick the value of the array
        }
    
    # Perform K-Means
    print('Performing K-Means')
    # Picking variables to X
    # Concatenate the data
    X = np.concatenate((np.array([data[key]['colors'][0] for key in data]), np.array([data[key]['dct'] for key in data])), axis=1)
    # Perform K-Means
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    # Get the labels
    labels = kmeans.labels_
    # Get the centroids
    centroids = kmeans.cluster_centers_
    # Get the inertia   
    inertia = kmeans.inertia_
    print('K-Means results:')
    print('Labels:', labels)
    print('Centroids:', centroids)
    print('Inertia:', inertia)

    # Print the label for each image
    # print('Printing the label for each image')
    # for key in data:
    #     print(f'Image {key} is in cluster {labels[int(key)] + 1}')

    # Plot the images in the clusters
    print('Saving the images in the clusters')
    for i in range(5):
        # Write a text file with the centroid
        with open(f'{global_variables.dir_museum}cluster_{i + 1}/centroid_info.txt', 'w') as f:
            f.write('Format is [B, G, R, First DCT Zig-Zag Coefficient]\n')
            f.write(f'Centroid {i + 1}: {centroids[i]}')
        for key in data:
            if labels[int(key)] == i:
                print('Saving image', key, 'in cluster', i + 1)
                # Add the image to the plot
                image = cv2.imread(global_variables.dir_db + f'bbdd_{key}.jpg')
                # Save the image in the cluster folder
                cv2.imwrite(global_variables.dir_museum + f'cluster_{i + 1}/{key}.jpg', image) 

def calculate_dct(image, f_name, dct):
    # Convert to grayscale
    patch_texture = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    X = 1 # Number of coefficients to consider

    m,n=patch_texture.shape
    if (m % 2) != 0:
        patch_texture = np.append(patch_texture, [np.zeros(n)], axis=0)
    m,n=patch_texture.shape
    if (n % 2) != 0:
        patch_texture = np.append(patch_texture, np.zeros((m,1)), axis=1)

    patch_float = np.float64(patch_texture)/255.0  
    patch_texture_dct = cv2.dct(patch_float)
    
    zigzag_vector = np.concatenate([np.diagonal(patch_texture_dct[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-patch_texture_dct.shape[0], patch_texture_dct.shape[0])])[:X]

    dct[f_name] = zigzag_vector

# Function: Get dominant color of the image using k-means clustering
def get_dominant_color(image, f_name, dominant_colors, k=3, image_processing_size = None):
    # resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, interpolation = cv2.INTER_AREA)
    # ! # convert to rgb from bgr
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # cluster the pixel intensities
    clt = KMeans(n_clusters = k)
    clt.fit(image)
    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    # create empty chart to be filled with bars
    # representing the relative frequency of each of the colors
    chart = np.zeros((50, 300, 3), np.uint8)
    start = 0
    # loop over the percentage of each cluster and the color of
    # Sort clt.cluster_centers_ and hist by the hist values
    clt.cluster_centers_ = clt.cluster_centers_[np.argsort(hist)][::-1]
    hist = np.sort(hist)[::-1]

    # Color saving
    dominant_colors[f_name] = {}
    dominant_colors[f_name]['colors'] = []
    dominant_colors[f_name]['percent'] = []

    # each cluster
    for (percent, color) in zip(hist, clt.cluster_centers_):
        # plot the relative percentage of each cluster
        end = start + (percent * 300)
        color_list = color.astype("uint8").tolist()
        cv2.rectangle(chart, (int(start), 0), (int(end), 50), color_list, -1)
        dominant_colors[f_name]['colors'].append(color_list)
        dominant_colors[f_name]['percent'].append(percent)
        start = end
    # return the bar chart
    return chart

if __name__ == "__main__":
    main_museum()