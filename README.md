This is a demo used for my Business IT Architecture class

This application utilizes real images of hand-written digits to train a convolutional neural network to make accurat predictions of the images' labels using Python. 

Packages used: keras, matplotlib, numpy.


THE DATA:

The dataset used to train this application is downloaded from http://yann.lecun.com/exdb/mnist/. It consists of 60,000 28x28 pixel images of hand-written digits that is used for training as "x", as well as 10,000 units for validations. Each image
contains a label as "y" to and tell us what the correct number displayed in the image is. 

The images were pre-treated by digitizing each pixel. To be more specific, an image is breaking down into 28 arrays with 28 cells, eaching containing a reading. The readings represense the brightness of each pixel with 0 being no writing in that 
cell and 255 being the brightest. 

The label contains a single digit of 0-9. We need to convert this data into an array since the input (28x28 pixels of readings) are in the format of arrays. It is easier to compare arrays with arrays rather than arrays with integers. This process is done by using the One-Hot method. The One-Hot method converts a number into an array by placing an "1" inside an array of 10 that consists only "0"s. The label can be identified by the location of the "1" in the array. For example, in [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.] the "1" is located at this 3rd index, which indecates that the label is a "2" (indexes starts at 0 in an array). Now, all the data is formatted correctly. We can begin the modeling process. 


THE MODEL:

The Keras Sequential model allow us to create layer-by-layer models for most of the common problems. Since the goal of this application is to use pre-treated digitized image data to make predictions of a single digit, thus, only one source of inputs and does not reuse layers, a Sequential model will be approriate to achieve the goal. 

