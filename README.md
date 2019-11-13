This is a demo used for my Business IT Architecture class

This application utilizes real images of hand-written digits to train a convolutional neural network to make accurate 
predictions of the images' labels using Python. 

Packages used: keras, matplotlib, numpy.


THE DATA:

The dataset used to train this application is downloaded from http://yann.lecun.com/exdb/mnist/. It consists of 60,000 
28x28 pixel images of hand-written digits that is used for training as well as 10,000 units for validations. Each image
contains a label and tell us the correct number displayed in the image. 

The images were pre-treated by digitizing each pixel. To be more specific, an image is breaking down into 28 arrays with
28 cells, eaching containing a reading. The readings represense the brightness of each pixel with 0 being no writing in that 
cell and 255 being the brightest. 

The label contains a single digit of 0-9. We need to convert this data into an array since the input (28x28 pixels of readings)
are in the format of arrays. It is easier to compare arrays with arrays rather than arrays with integers. This process is done
by using the One-Hot method. It con
