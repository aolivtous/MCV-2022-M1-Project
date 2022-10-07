"""
    Main file of the project
"""

# External modules
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.ma.core import mean
import pickle

# Internal modules
import histograms

base_dir = "../"

def main():
    print('Main execution')
    # # read image 
    # # imageLab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # image = cv2.imread(base_dir + 'BBDD/bbdd_00000.jpg')
    # # show the image, provide window name first
    # cv2.imshow('image', image)
    # # add wait key. window waits until user presses a key
    # cv2.waitKey(0)
    # # and finally destroy/close all open windows
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
