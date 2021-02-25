# Semicon-wafer-classification

Here I use a convolutional neural net to classify semiconductor wafers by one of 8 fault types. The NN achieves an overall accuracy of about 88% using 40 epochs. More epochs will top the models out at 92% but with considerable overtraining.

The semicon wafers come in various size matrices, and have no depth. They are each characterized by one of eight fault types, and the pixel size ranges from about 100 to over 3000, so a few steps had to be taken to create an effective model.

The first was to eliminate matrices that were too small or too large, as these would not resize well. I also created a function to remove noise from the wafers, though I may have overdone it. Once these steps were taken, i used cv2 to resize all usable images to 40x40 pixels, or elements.

From here, I trained a simple but effective convolutional neural net to classify the wafers. I used 4 different samples for training, with different resize methods and with some having gone through the noise reducing function I built.

There was a paper done by some students at Stanford, using the same dataset, wherein they used more state of the art CNN's. The paper is here:

http://cs230.stanford.edu/projects_fall_2019/posters/26259758.pdf

Despite using more sophisticated NN's, they did not truly outperfom the models i've created here, and in fact my models outperform there models when considering prediction of the "local" fault type. This is primarily because i made sure to augment my sample base to have an even number of each fault type.

While one of their models produces 92% testing accuracy, it is only managed with certain overtraining, as the training accuracy in that instance hit nearly 100%.

Given all this, I feel reasonably confident this model is near to complete optimization.

The initial wafer pickle required to run the semicon_p1.ipynb notebook can be downloaded here:

https://drive.google.com/drive/folders/1b8WBaw4QhCtq8IEeodpqg2P_UMVSQ4sB?usp=sharing
