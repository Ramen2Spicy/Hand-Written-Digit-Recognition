This is a demo used for my Business IT Architecture class

This application utilizes real images of hand-written digits to train a convolutional neural network to make accurate predictions of single digits contained in an image using Python. 

Packages used: keras, matplotlib, numpy.


PREPARING THE DATA:

The dataset used to train this application is downloaded from http://yann.lecun.com/exdb/mnist/. It consists of 60,000 28x28 pixel images of hand-written digits that is used for training as "x", as well as 10,000 units for validations. Each image
contains a label as "y" to tell us what the correct number displayed in the image is. 

The images were pre-treated by digitizing each pixel. To be specific, an image is breaking down into 28 arrays with 28 cells, each containing a reading. The readings represent the brightness of each pixel with 0 being no writing in that 
cell and 255 being the brightest. The brightness determines the significance of each pixel as well as helps us identify edges and patterns, we need to take them into an account. The data by default, has two dimensions, height and weights as represented by the number of rows and columns. The data must be reshaped by assigning a value for the third dimension or the depth - brightness. We can do so by typing X_train.reshape(60000, 28, 28, 1), notice the “60000” indicates the size of data reshaping and the "1" indicates that there is only one color (if dealing with colored image, the depth should be 3 for 3 RGB values). We now have 60,000 units of data with 3 dimensions (location on the x-axis, location on the y-axis, and brightness)

The label contains a single digit of 0-9. We need to convert this data into an array since the input (28x28 pixels of readings) are in the format of arrays. It is easier to compare arrays with arrays rather than arrays with integers. This process is done by using the One-Hot method. The One-Hot method converts a number into an array by placing an "1" inside an array of 10 that consists only "0"s. The label can be identified by the location of the "1" in the array. For example, in [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.] the "1" is located at this 3rd index, which indicates that the label is a "2" (indexes starts at 0 in an array). Now, all the data is formatted correctly. We can begin the modeling process. 


ESTABLISHING THE MODEL:

The Keras Sequential model allow us to create layer-by-layer models for most of the common problems. Since the goal of this application is to use pre-treated digitized image data to make predictions of a single digit, thus, the process is linear convolutional with only one source of inputs and does not reuse layers, a Sequential model will be appropriate to achieve the goal. Next, we add layers in the neural network. Since the inputs in this application, consists of only images, we will be using Conv2D as our layer type. To properly configure the layer, Conv2D requires certain arguments, filters, kernel size, activation, and input shape. 

A filter is a matrix of numbers that corresponds to certain patterns or features that sometimes, humans are incapable of detecting. We assign 60 filters in the first layer, which will produce 60 results, thus 60 inputs for the next layer. The 60 filters are designed to allow the application to pick up edges, the 30 filters assigned in the next layer will use the edges detected to form circles, curvatures, or straight lines. Read more about filters here: https://www.saama.com/blog/different-kinds-convolutional-filters/

Kernel of SVM (Support Vector Machine) is a commonly used method for classification that ascend the mapping the non-linear separable dataset into a higher dimensional space where we can find a hyperplane that can separate the samples. We will use kernel_size = 3 to configure the application to use data from all 3 dimensions. You can read more about kernels and SVM here: https://towardsdatascience.com/truly-understanding-the-kernel-trick-1aeb11560769. 

The input shape is the data dimensions that the layer should take account for; in this case, we take height, width, and depth (brightness) into the convolutive computation. As configured in the code, enter input_shape = (28, 28,1) for the dimension arguments.

Activation is a mathematic function that translate the result of a node into a binary format with x!=0 being activated and x=0 being “not activated”. The ReLu, Rectified Linear Unit function converts any number less than 0 into a 0, and keeps anything greater than 0 the way it is. Activation helps us to determine which patterns or features detected are significant enough to pass to the next layer. 

Again, our goal is to use a 3-dimensional data, location on x-axis, location to y-axis, and brightness to predict one single digit, as the name of the layer, the flattern layer will do the job. A flattern layer flatterns the inputs by breaking down the multi-dimensional data into one single array that we use as inputs for the next dense layer. This step is essential because the dense layer only accepts one signle array as its imputs. 

The dense layer enforces a connection between each input and output, by which the ouyput is weighted. By useing an activation type "Softmax" in a dense layer, each output is normalized into a number between 0 to 1 in the corresponding idex of an array.


COMPILING:

Lastly, we need to compile the code by specifying the optimizer, loss, and metrics. We will be using the "ADAM" (Adaptive Movement Estimation) optimizer to optimize the gradient desent of the model. Read more about gradient desent here: https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e. This process will help the nerual network to find the best adjustment for each weights and biases after processing every image. 

The loss function is a method that describes the performance of the model. Because of the outputs of the nerual network are  uninterpreted and the nature of classification (predictions are either right or wrong), the loss function allows us to see how close the calcualtions that lead to the final decisions are. Note that this is different from accuracy since the loss function calculates how close the proccess of predictions are to the labels, insteand of calculating how close the end results are to the labels.

The actuall accuracy is where we compare the end results to the label are specified by "metrics".


FITTING THE MODEL:

In the field of Machine Learning, fitting generally means to train. We have prepared the data into the correct format and set model configurations, it's time to feed the data to the model. We can do so by identifying the training data and validation data(optional), in this case, x_train, y_train, and x_test, y_test. During the fitting process, the model randomly generates weights and biases and compare its predicted result with the image label. Based on the accuracy, the model re-adjusts the weights and biases. Similar to conditioning human behaviors through a reinforce-punish system, if the predicted result was correct, the model will reward itself by adjusting the specific weights and biases that lead to the correct prediction more, and vice versa. This process enables the model to process a large amount of data to self-train with an aim to reach higher accuracy.

Epochs essentially determine how many times the model will go through the training dataset. The model might not pick up patterns that are significant during the early training stage. By going through the dataset multiple times, it assures the model to capitalize all training data containing in the dataset. 

MAKING PREDICTIONS

The Keras_predict method allows us to make prediction with the trained model. The predicted results, however, are still generated in the format of arrays. For example, in an array of [2.69685412e-08 6.92543023e-10 1.87893279e-09 9.99994040e-01
  1.34154798e-09 1.40939324e-06 7.01640843e-15 3.29851944e-11. 4.28513931e-06 2.42023674e-07], each number in the index indicates the possibility that a "1" is located at the corresponding index. In another word, there is 9.99994040e-01 chance that there is a "1" located in the 4th index. Therefore the model predicts that there is a 9.99994040e-01 chance this image contains a "3". This format of result is still hard to read, however, we could use the numpy.argmax function to translate the result directly into the image label. 
