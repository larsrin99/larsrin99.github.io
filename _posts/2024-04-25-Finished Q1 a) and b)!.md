# Q1 a) and b) in Assignment 2
## After a long process I have finally been able to enroll and compare fingerprints.
The GUI takes in a pathfile of an image, then puts it through the process functions.
The process functions fetches the image, valid minutiae and local structures, and returns them to either be enrolled or compared.

![Image of GUI, with outputs below](/images/Correct_output.png)

In this image, we have the whole GUI and the outputs from entering three files and then comparing a new fourth.
The comparing function runs our compare_fingers function where it compares our input files local structures ls1, to every local structure in our pickle file.

The pickle files stores image, valid minutiae and local structures with the filename as a key.
So when running the comparing function it will find the best possible match and print both the score and the filename of the best match.
