import compare_prints
import processing_prints


def evaluate_system(input_fingerprint, database_fingerprints, threshold):
    # Initialize counts for true positive, true negative, false positive, and false negative
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Iterate through database fingerprints
    for fingerprint in database_fingerprints:
        # Compare input fingerprint with each fingerprint in the database
        similarity_score = compare_prints.compare_fingers(input_fingerprint, fingerprint)

        # Check if similarity score is above threshold
        if similarity_score > threshold:
            # Compare with ground truth (filename)
            if fingerprint.filename == input_fingerprint.filename:
                # True positive: Correctly identified as the same person
                TP += 1
            else:
                # False positive: Incorrectly identified as the same person
                FP += 1
        else:
            # If similarity score is below threshold, it's not considered a match
            # Check if ground truth confirms this (filename)
            if fingerprint.filename != input_fingerprint.filename:
                # True negative: Correctly identified as different persons
                TN += 1
            else:
                # False negative: Incorrectly identified as different persons
                FN += 1

    # Construct confusion matrix
    confusion_matrix = [[TP, FP], [FN, TN]]