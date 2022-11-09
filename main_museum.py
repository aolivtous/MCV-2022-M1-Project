import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

# Internal modules
import global_variables

def main_museum():
    for filename in tqdm(os.scandir(global_variables.dir_db)):
        f = os.path.join(global_variables.dir_db, filename)
        # checking if it is a file
        if f.endswith('.jpg'):
            f_name = filename.name.split('.')[0].split('_')[1]
            image = cv2.imread(f)
            chart = get_dominant_color(image)
            cv2.imwrite(global_variables.dir_museum + f_name + '_dominant_color.jpg', chart)

# Function: Get dominant color of the image using k-means clustering
def get_dominant_color(image, k=4, image_processing_size = None):
    # resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, interpolation = cv2.INTER_AREA)
    # convert to rgb from bgr
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    # each cluster
    for (percent, color) in zip(hist, clt.cluster_centers_):
        # plot the relative percentage of each cluster
        end = start + (percent * 300)
        cv2.rectangle(chart, (int(start), 0), (int(end), 50),
            color.astype("uint8").tolist(), -1)
        start = end
    # return the bar chart
    return chart
            

if __name__ == "__main__":
    main_museum()