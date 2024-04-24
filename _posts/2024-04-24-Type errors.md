Currently I am facing issues when trying to load enrolled prints from the pickle file.
I believe the issue is in the way it is stored, and occured when I tried to compare a print by iterating the pickle file.

------ UPDATE! ------

I seem to run into the same error:

![Type Error in pickle](/images/Error_message_GUI1.png)

My GUI seems to be able to enroll fingerprints with no issue, but when comparing I run into trouble with loading files from pickle.
I want my enrollments to be stored with the tags: fingerprint, valid_minutiae, local_structures.
And then using the local_structures to compare new prints to the existing ones. I also want it to only output the best comparison score along with the corresponding fingerprint file:


    def compare_fingers(input_file_path):
        # Load enrolled features from the pickle file
        with open('enrolled_features.pickle', 'rb') as f:
            enrolled_features_list = pickle.load(f)
            
        # Process input fingerprint to extract features
        f1, m1, ls1 = processing_prints.process_and_extract_features(file_path)
    
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


        
I also updated the display of my GUI!

![New GUI (Again..)](/images/NewGUI_16.30.png)

Blogging is a bit more fun now:)
