Finally started on the fingerprint tasks.

Never coded a GUI, but after a couple of youtube tutorials and some help from chatGPT I have created something basic:


```

import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np

class FingerprintRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerprint Recognition System")

        self.enrolled_fingerprints = {}

        self.create_widgets()

    def create_widgets(self):
        # Enrollment Frame
        enrollment_frame = tk.Frame(self.root, padx=10, pady=10)
        enrollment_frame.grid(row=0, column=0, padx=10, pady=10)

        tk.Label(enrollment_frame, text="Enroll Fingerprint:").grid(row=0, column=0, columnspan=2)

        self.enroll_button = tk.Button(enrollment_frame, text="Enroll", command=self.enroll_fingerprint)
        self.enroll_button.grid(row=1, column=0, padx=5, pady=5)

        # Comparison Frame
        comparison_frame = tk.Frame(self.root, padx=10, pady=10)
        comparison_frame.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(comparison_frame, text="Compare Fingerprint:").grid(row=0, column=0, columnspan=2)

        self.compare_button = tk.Button(comparison_frame, text="Compare", command=self.compare_fingerprint)
        self.compare_button.grid(row=1, column=0, padx=5, pady=5)

        # Evaluation Frame
        evaluation_frame = tk.Frame(self.root, padx=10, pady=10)
        evaluation_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        tk.Label(evaluation_frame, text="Evaluation:").grid(row=0, column=0, columnspan=2)

        self.evaluate_button = tk.Button(evaluation_frame, text="Evaluate", command=self.evaluate_system)
        self.evaluate_button.grid(row=1, column=0, padx=5, pady=5)

    def enroll_fingerprint(self):
        # Placeholder function for enrolling fingerprints
        messagebox.showinfo("Enroll Fingerprint", "Fingerprint enrolled successfully.")

    def compare_fingerprint(self):
        # Placeholder function for comparing fingerprints
        messagebox.showinfo("Compare Fingerprint", "Fingerprint compared successfully.")

    def evaluate_system(self):
        # Placeholder function for evaluating system
        messagebox.showinfo("Evaluate System", "System evaluation completed.")

def main():
    root = tk.Tk()
    app = FingerprintRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
```

![Image of GUI](/images/GUI_draft1.png)
