import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)
        # TODO Create necessary layers
#         raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        for _, param in self.params().items():
            param.grad = np.zeros(param.grad.shape)
        # Hint: using self.params() might be useful!
#         raise Exception("Not implemented!")
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        results = self.fc1.forward(X)
        results = self.relu.forward(results)
        results = self.fc2.forward(results)
        loss, d_predictions = softmax_with_cross_entropy(results, y)
#         print("loss before regularization =", loss)
#         grad_reg = 0
#             print("Total grad = ", param.grad)
#         print("Reg loss for all parameters =", loss_reg)
        d_input_fc2 = self.fc2.backward(d_predictions)
        d_input_relu = self.relu.backward(d_input_fc2)
        d_input_fc1 = self.fc1.backward(d_input_relu)
        
        loss_reg = 0

        for param_name, param in self.params().items():
            loss_param, grad = l2_regularization(param.value, self.reg)
#             print(f"reg_loss for {param_name} = ", loss_param)
            loss_reg += loss_param
            param.grad += grad
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
#         raise Exception("Not implemented!")

        return loss + loss_reg

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        results = self.fc1.forward(X)
        results = self.relu.forward(results)
        results = self.fc2.forward(results)
        
        probs = results - np.max(results, axis=1, keepdims=True)
        probs = np.exp(probs) 
        probs /= np.sum(probs, axis=1, keepdims=True)

        pred = np.argmax(probs, axis=1)

#         raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {'W1': self.fc1.W, 'B1': self.fc1.B,
                  'W2': self.fc2.W, 'B2': self.fc2.B}

        # TODO Implement aggregating all of the params

#         raise Exception("Not implemented!")

        return result
