import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    
    loss = np.sum(W * W) * reg_strength
    grad = 2 * reg_strength * W
#     raise Exception("Not implemented!")
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    
    batch_size = preds.shape[0]
#     if batch_size == 0:
#         print("WTF")
#     print("batch_size = ", batch_size)
    target_probs = np.zeros(preds.shape)
    target_probs[range(batch_size), target_index] = 1
    
    probs = preds - np.max(preds, axis=1, keepdims=True)
    probs = np.exp(probs) 
    probs /= np.sum(probs, axis=-1, keepdims=True)
    
#     return np.exp(predictions - max_pred) / np.sum(np.exp(predictions - max_pred), axis=-1, keepdims=True)    
#     probs = np.exp(preds - np.max(preds))/np.exp(preds).sum()
#     loss = -1/batch_size * (target_probs * np.log(probs)).sum()
#     if batch_size == 0:
#         print("WTF")
#     loss = -np.log(np.choose(target_index, probs.T)).mean()
    loss = - (np.log(probs) * target_probs).sum()/batch_size

    dprediction = (probs - target_probs)/batch_size
#     raise Exception("Not implemented!")

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.relu_mask = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
#         raise Exception("Not implemented!")
#         print("HI from forward-pass, x=\n", str(X))    
#         print("X before RELU: \n", str(X))
        self.relu_mask = X<0
#         print("HI from forward-pass " + str(self.relu_mask.shape))
        X = X.copy()
        X[self.relu_mask]=0
#         print("X after relu: \n" + str(X))
        return X

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
#         raise Exception("Not implemented!")
#         print("HI from backward-pass " + str(d_out.shape))
        d_result = d_out    
        d_result[self.relu_mask] = 0
#         print(d_out)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
#         print("HI from layer step forward")
        self.X = X
#         print("shape of X: ", str(X.shape))
#         print("shape of W: ", str(self.W.value.shape))
        results = np.dot(self.X, self.W.value) + self.B.value
#         print("shape of results after X * W + B: ", str(results.shape))
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
#         raise Exception("Not implemented!")
        return results

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

#         loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
        batch_size = d_out.shape[0]
#         print("HI from back prop step!, shape of d_out = ", d_out.shape)
        self.W.grad = np.dot(self.X.T, d_out) #3x2, 2x4 #1/batch_size * 
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)
        d_input = np.dot(d_out, self.W.value.T)
        
#         print("Shape of d_input = ", d_input.shape)

        
        
        # It should be pretty similar to linear classifier from
        # the previous assignment

#         raise Exception("Not implemented!")

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
