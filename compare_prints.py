import os
import pickle
import processing_prints
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *

def compare_fingers(input_file_path, enrolled_features_list):
   # Process input fingerprint to extract features
    f1, m1, ls1 = processing_prints.process_and_extract_features(input_file_path)

    best_match_index = None
    best_score = -float('inf')

    # Iterate over each set of enrolled features
    for idx, enrolled_features in enumerate(enrolled_features_list):
        # Extract features
        f2 = enrolled_features['fingerprint_image']
        m2 = enrolled_features['valid_minutiae']
        ls2 = enrolled_features['local_structures']

        # Resize ls1 to match the shape of ls2
        #ls1_resized = cv.resize(ls1, (ls2.shape[1], ls2.shape[0]))
        
        dists = np.linalg.norm(ls1[:,np.newaxis,:] - ls2, axis = -1)
        dists /= np.linalg.norm(ls1, axis = 1)[:,np.newaxis] + np.linalg.norm(ls2, axis = 1) # Normalize as in eq. (17) of MCC paper
        
        # Select the num_p pairs with the smallest distances (LSS technique)
        num_p = 5 # For simplicity: a fixed number of pairs
        pairs = np.unravel_index(np.argpartition(dists, num_p, None)[:num_p], dists.shape)
        score = 1 - np.mean(dists[pairs[0], pairs[1]])

        # Update best match if needed
        if score > best_score:
            best_match_index = idx
            best_score = score

   # Print out the best match score, and filename of the best match
    best_match_features = enrolled_features_list[best_match_index]
    print(f"Best match score: {best_score}")
    print(f"Filename of the best match: {best_match_features['filename']}")
    return best_score

