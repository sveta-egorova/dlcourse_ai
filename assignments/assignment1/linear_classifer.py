import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
#     print("pred\n" + str(predictions))
#     print("max predictions: \n" + str(np.max(predictions, axis=1)))
    probs = predictions - np.atleast_2d(np.max(predictions, axis=1)).T
#     print("probs\n" + str(probs))
    probs = np.exp(probs) 
#     print("probs\n" + str(probs))
    probs /= np.atleast_2d(probs.sum(axis=1)).T
#     print("probs\n" + str(probs))

    # TODO implement softmax
    # Your final implementation shouldn't have any loops
#     raise Exception("Not implemented!")
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    
    batch_size = target_index.shape[0]

#     print("target index = \n", target_index)
    target_probs = np.zeros(probs.shape)
    target_probs[range(batch_size), target_index] = 1
#     for i in range(batch_size):
#         target_probs[i,target_index[i]] = 1
#     print("target_probs = \n", str(target_probs))
    
#     print("batch_size =" + str(batch_size))
#     print("probs[:, target_index] = " + str(probs[:, target_index]))
#     print("LN OF PROBS = \n", str(np.log(probs)))
#     print("ALMOST LOSS = \n", str(np.log(probs) * target_probs))
    losses = - (np.log(probs) * target_probs).sum()/batch_size
#     losses = losses.sum()/batch_size
#     print("losses = ", str(losses))
#     losses[target_index]
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
#     raise Exception("Not implemented!")
    return losses


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    
    batch_size = target_index.shape[0]
    target_probs = np.zeros(predictions.shape)
    target_probs[range(batch_size), target_index] = 1
#     for i in range(batch_size):
#         target_probs[i,target_index[i]] = 1
    
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = (probs - target_probs)/batch_size
#     print("softmax of predictions = " + str(dprediction))
#     print("target index = " + str(target_index))
#     dprediction[:,target_index] -= 1
#     print("after deducting 1 = " + str(dprediction))
#     print("Done with softmax_CE function")
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
#     raise Exception("Not implemented!")

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = np.multiply(W, W).sum() * reg_strength
    grad = 2 * reg_strength * W
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
#     raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    batch_size = target_index.shape[0]
    predictions = np.dot(X, W)
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction) # (num_features, num_batch) x (batch_size, N) 
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
#     raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            batch = X[batches_indices, :]
            loss_linear, dW = linear_softmax(X, self.W, y)
            loss_reg, dW_reg = l2_regularization(self.W, reg)
            self.W -= (dW + dW_reg) * learning_rate
            loss = loss_linear + loss_reg
            loss_history.append(loss)
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
#             raise Exception("Not implemented!")

            # end
#             print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        prob = softmax(np.dot(X, self.W))
        y_pred = np.argmax(prob, axis=1)
        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
#         raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
