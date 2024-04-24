import os
import pickle
import processing_prints
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *

def compare_fingers(input_file_path):
    # Load enrolled features from the pickle file
    with open('enrolled_features.pickle', 'rb') as f:
        enrolled_features_list = pickle.load(f)

    # Process input fingerprint to extract features
    f1, m1, ls1 = processing_prints.process_and_extract_features(input_file_path)

    comparison_scores = []
    best_match_index = None
    best_score = -float('inf')

    # Iterate over each set of enrolled features
    for idx, enrolled_features in enumerate(enrolled_features_list):
        # Extract features
        f2 = enrolled_features['fingerprint_image']
        m2 = enrolled_features['valid_minutiae']
        ls2 = enrolled_features['local_structures']

        # Compute comparison score between ls1 and ls2
        # For example, compute the mean absolute error or any other similarity/distance metric
        score = np.mean(np.abs(ls1 - ls2))

        # Update best match if needed
        if score > best_score:
            best_match_index = idx
            best_score = score

        # Append the comparison score to the list
        comparison_scores.append(score)

    # Calculate the overall comparison score as the average of all individual scores
    overall_score = np.mean(comparison_scores)

    # Print out the best match index, score, and filename of the best match
    best_match_features = enrolled_features_list[best_match_index]
    print(f"Best match index: {best_match_index}")
    print(f"Best match score: {best_score}")
    print(f"Filename of the best match: {best_match_features['filename']}")