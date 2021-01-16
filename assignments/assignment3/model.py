import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.filter_size = 3
        self.pool_size = 4
        self.padding = 1    # input shape: m x 32 x 32 x 3
        self.conv1 = ConvolutionalLayer(input_shape[2], conv1_channels, self.filter_size, self.padding) # shape: m x 32 x 32 x conv1_channels
        self.relu1 = ReLULayer() # shape: m x 32 x 32 x conv1_channels
        self.maxpool1 = MaxPoolingLayer(self.pool_size, self.pool_size) # shape m x 8 x 8 x conv1_channels
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, self.filter_size, self.padding) # shape m x 8 x 8 x conv2_channels
        self.relu2 = ReLULayer() # shape m x 8 x 8 x conv2_channels
        self.maxpool2 = MaxPoolingLayer(self.pool_size, self.pool_size) # shape m x 2 x 2 x conv2_channels
        self.flat = Flattener() # shape m x 4 * conv2_channels
        self.fc = FullyConnectedLayer(4 * conv2_channels, n_output_classes)
        
#         raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        for _, param in self.params().items():
            param.grad = np.zeros(param.grad.shape)
        
        # TODO Compute loss and fill param gradients
        
        results = self.conv1.forward(X)
        results = self.relu1.forward(results)
        results = self.maxpool1.forward(results)
        results = self.conv2.forward(results)
        results = self.relu2.forward(results)
        results = self.maxpool2.forward(results)
        results = self.flat.forward(results)
        results = self.fc.forward(results)
        loss, d_predictions = softmax_with_cross_entropy(results, y)
        
        d_input = self.fc.backward(d_predictions)
        d_input = self.flat.backward(d_input)
        d_input = self.maxpool2.backward(d_input)
        d_input = self.relu2.backward(d_input)
        d_input = self.conv2.backward(d_input)
        d_input = self.maxpool1.backward(d_input)
        d_input = self.relu1.backward(d_input)
        d_input = self.conv1.backward(d_input)

        
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        return loss
#         raise Exception("Not implemented!")

    def predict(self, X):
        # You can probably copy the code from previous assignment
#         pred = np.zeros(X.shape[0], np.int)
        results = self.conv1.forward(X)
        results = self.relu1.forward(results)
        results = self.maxpool1.forward(results)
        results = self.conv2.forward(results)
        results = self.relu2.forward(results)
        results = self.maxpool2.forward(results)
        results = self.flat.forward(results)
        results = self.fc.forward(results)
        
        probs = results - np.max(results, axis=1, keepdims=True)
        probs = np.exp(probs) 
        probs /= np.sum(probs, axis=1, keepdims=True)

        pred = np.argmax(probs, axis=1)
        return pred
#         raise Exception("Not implemented!")

    def params(self):
        result = {'conv1_W': self.conv1.W, 
                  'conv1_B': self.conv1.B, 
                  'conv2_W': self.conv2.W, 
                  'conv2_B': self.conv2.B,
                  'fc_W': self.fc.W, 
                  'fc_B': self.fc.B
                 }

        # TODO: Aggregate all the params from all the layers
        # which have parameters
#         raise Exception("Not implemented!")

        return result
