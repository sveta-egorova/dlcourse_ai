import numpy as np


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
    # TODO: Copy from previous assignment
#     raise Exception("Not implemented!")
    loss = np.sum(W * W) * reg_strength
    grad = 2 * reg_strength * W

    return loss, grad


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
    # TODO copy from the previous assignment
#     raise Exception("Not implemented!")
    batch_size = predictions.shape[0]

    target_probs = np.zeros(predictions.shape)
    target_probs[range(batch_size), target_index] = 1
    
    probs = predictions - np.max(predictions, axis=1, keepdims=True)
    probs = np.exp(probs) 
    probs /= np.sum(probs, axis=-1, keepdims=True)

    loss = - (np.log(probs) * target_probs).sum()/batch_size
#     loss = -np.log(np.choose(target_index, probs.T)).mean() # need to check
    dprediction = (probs - target_probs)/batch_size
    
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.relu_mask = None


    def forward(self, X):
        # TODO copy from the previous assignment
#         raise Exception("Not implemented!")
        self.relu_mask = X<0
        X = X.copy()
        X[self.relu_mask]=0
        return X
#         return np.maximum(X, np.zeros_like(X)) # need to check

    def backward(self, d_out):
        # TODO copy from the previous assignment
#         raise Exception("Not implemented!")
        d_result = d_out    
        d_result[self.relu_mask] = 0
        return d_result    

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
#         raise Exception("Not implemented!")
        self.X = X.copy()
        results = np.dot(self.X, self.W.value) + self.B.value
        return results

    def backward(self, d_out):
        # TODO copy from the previous assignment
#         raise Exception("Not implemented!")
        self.W.grad += np.dot(self.X.T, d_out) 
        self.B.grad += np.sum(d_out, axis=0, keepdims=True)
        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
    
class ConvolutionalLayer1:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        X_padded = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, self.in_channels))
        X_padded[:, self.padding:self.padding + height, self.padding:self.padding + width, :] = X
        self.X_cache = (X, X_padded)
        X_padded = X_padded[:, :, :, :, np.newaxis]

        W = self.W.value[np.newaxis, :, :, :, :] # Why use np.newaxis??????   [smth, filter x filter_size x in_channels, out_channels]

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                X_slice = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :, :]    # shape [batch, filter, filter, in_channels, smth]
                out[:, y, x, :] = np.sum(X_slice * self.W.value, axis=(1, 2, 3)) + self.B.value # WTF?????

        return out

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        X, X_padded = self.X_cache

        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        X_grad = np.zeros_like(X_padded)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)

                X_slice = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :, np.newaxis]
                grad = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :] # WTF?????
                self.W.grad += np.sum(grad * X_slice, axis=0) # WTF?????

                X_grad[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.sum(self.W.value * grad, axis=-1)

        self.B.grad += np.sum(d_out, axis=(0, 1, 2))

        return X_grad[:, self.padding:self.padding + height, self.padding:self.padding + width, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }
    
    
    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.X = None
        self.padding = padding


    def forward(self, X):
        self.X = X

        batch_size, height, width, channels = self.X.shape

        out_height = height + 2 * self.padding - self.filter_size + 1
        out_width = width + 2 * self.padding - self.filter_size + 1
        output = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        if self.padding > 0:
            h_padding = np.zeros((batch_size, self.padding, width, channels))
            self.X = np.concatenate((h_padding, self.X, h_padding), axis=1) # add padding horizontally
            v_padding = np.zeros((batch_size, height + 2 * self.padding, self.padding, channels)) # add padding vertically
            self.X = np.concatenate((v_padding, self.X, v_padding), axis=2)
        
        for y in range(out_height):
            for x in range(out_width):
                h1 = y
                h2 = y + self.filter_size
                w1 = x
                w2 = x + self.filter_size
                
                X_conv = self.X[:, h1:h2, w1:w2, :]  # shape [batch, filter, filter, in_channels]
                X_conv_flat = X_conv.reshape(X_conv.shape[0], -1) # shape [batch, filter x filter x in_channels]
                W_flat = self.W.value.reshape(-1, self.W.value.shape[-1]) # shape [filter x filter_size x in_channels, out_channels]
                output[:, y, x, :] = np.dot(X_conv_flat, W_flat) + self.B.value # shape [batch, out_channels]
                
                # TODO: Implement forward pass for specific location
#                 pass
#         raise Exception("Not implemented!")
        return output

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

#         print("CONV: d_out_shape = ", d_out.shape)

        
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape # batch x conv_height x conv_weight x out_channels
        
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        d_input = np.zeros(self.X.shape)
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                h1 = y
                h2 = y + self.filter_size
                w1 = x
                w2 = x + self.filter_size
                
                X_orig_piece = self.X[:, h1:h2, w1:w2, :] # shape [batch, filter, filter, in_channels]
                X_orig_flat = X_orig_piece.reshape(X_orig_piece.shape[0], -1) # shape [batch, filter x filter x in_channels]
                d_out_conv = d_out[:, y, x, :] # shape [batch x out_channels]
                self.W.grad += np.dot(X_orig_flat.T, d_out_conv).reshape(
                    self.filter_size, self.filter_size, self.in_channels, self.out_channels, order = 'C')
                self.B.grad += np.sum(d_out_conv, axis=0)
                W_flattened = self.W.value.reshape(-1, self.W.value.shape[-1]) #shape [filter x filter x in_channels, out_channels]
                d_input_backprop = np.dot(d_out_conv, W_flattened.T) # shape [batch, filter x filter x in_channels]
                d_input_backprop = d_input_backprop.reshape(
                    d_input_backprop.shape[0], self.filter_size, self.filter_size, self.in_channels, order = 'C')
                d_input[:, h1:h2, w1:w2, :] += d_input_backprop
                
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)

#         raise Exception("Not implemented!")
        if self.padding > 0:
            # remove horizontal padding 
            d_input = d_input[:, self.padding : -self.padding, :, :]
            # remove vertical padding 
            d_input = d_input[:, :, self.padding : -self.padding, :]

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.pool_index = None

    def forward(self, X):
        self.X = X

        batch_size, height, width, channels = self.X.shape
        
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        out_height = int((height - self.pool_size)/self.stride + 1)
        out_width = int((width - self.pool_size)/self.stride + 1)
        output = np.zeros((batch_size, out_height, out_width, channels))
        
#         self.pool_index_dim1 = np.zeros((out_height, out_width, batch_size * channels)).astype(int)
#         self.pool_index_dim2 = np.zeros((out_height, out_width, batch_size * channels)).astype(int)
#         self.pool_index_dim3 = np.zeros((out_height, out_width, batch_size * channels)).astype(int)
        
        for y in range(out_height):
            for x in range(out_width):
                h1 = y
                h2 = y + self.pool_size
                w1 = x
                w2 = x + self.pool_size
                X_piece = self.X[:, h1:h2, w1:w2, :]
                output[:, y, x, :] = np.amax(X_piece, axis=(1, 2))
#                 output[:, y, x, :] = X_piece.max(axis=(1,2))

#                 X_piece_3D = X_piece.reshape(X_piece.shape[0], X_piece.shape[2], -1)
#                 X_piece_3D_max_index = np.argmax(X_piece_3D, axis=2)
#                 self.pool_index_dim1[y,x,:] = np.arange(X_piece_3D.shape[0]).repeat(X_piece_3D.shape[1])
#                 self.pool_index_dim2[y,x,:] = np.tile(np.arange(X_piece_3D.shape[1]), (X_piece_3D.shape[0]))
#                 self.pool_index_dim3[y,x,:] = X_piece_3D_max_index.reshape(-1)

#                 max_values = X_piece_3D[self.pool_index_dim1[y,x,:], 
#                                         self.pool_index_dim2[y,x,:], 
#                                         self.pool_index_dim3[y,x,:]].reshape(X_piece_3D.shape[0], X_piece_3D.shape[1])
#                 output[:, y, x, :] = max_values

#                 output[:, y, x, :] = X_piece_3D[self.pool_index_dim1[y,x,:], 
#                                                 self.pool_index_dim2[y,x,:],
#                                                 self.pool_index_dim3[y,x,:]]
                                
        return output
#         raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, channels = d_out.shape 
        d_input = np.zeros(self.X.shape) # G
#         d_input_3D = np.zeros(self.X.shape[0], -1, self.X.shape[2])  # B 
        
        for y in range(out_height):
            for x in range(out_width):
                h1 = y
                h2 = y + self.pool_size
                w1 = x
                w2 = x + self.pool_size
#                 X_orig_piece = self.X[:, h1:h2, w1:w2, :] # shape [batch, pool, pool, channel]
                X_slice = self.X[:, h1:h2, w1:w2, :]
                grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                mask = (X_slice == np.amax(X_slice, (1, 2))[:, np.newaxis, np.newaxis, :])
                d_input[:, h1:h2, w1:w2, :] += grad * mask            
    
#                 d_input_piece = d_input[:, h1:h2, w1:w2, :]            
#                 d_out_piece_3D = d_out[:, y, x, :].reshape(-1) # A
#                 d_input_3D = np.zeros((batch_size, channels, self.pool_size*self.pool_size)) # B
#                 d_input_3D[self.pool_index_dim1[y,x,:], 
#                            self.pool_index_dim2[y,x,:],
#                            self.pool_index_dim3[y,x,:]] = d_out_piece_3D
#                 d_input_4D_updated = d_input_3D.reshape(d_input_piece.shape)
#                 d_input[:, h1:h2, w1:w2, :] += d_input_4D_updated
            
#                 d_input_piece_3D = d_input_piece.reshape(d_input_piece.shape[0], d_input_piece.shape[2], -1)
#                 d_input_to_update = d_input_piece_3D[self.pool_index_dim1[y,x,:], 
#                                                      self.pool_index_dim2[y,x,:],
#                                                      self.pool_index_dim3[y,x,:]].reshape(d_input_piece_3D.shape[0],
#                                                                                           d_input_piece_3D.shape[1])
                
#                 d_input_to_update += d_out_piece
#                 d_input[:, h1:h2, w1:w2, :] = d_input_to_update
                
                
#                 for ex in range(batch_size):
#                     for ch in range(channels):
#                         index_max_value = np.unravel_index(X_orig_piece[ex, :, :, ch].argmax(), X_orig_piece[ex, :, :, ch].shape)
#                         d_input[ex, :, :, ch][index_max_value] += d_out[ex, y, x, ch] 

        return d_input
#         raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = self.X.shape
        
        result = self.X.reshape(self.X.shape[0], -1)

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
#         raise Exception("Not implemented!")
        return result

    def backward(self, d_out):
#         print("FLAT: d_out_shape = ", d_out.shape)

        # TODO: Implement backward pass
        d_input = d_out.reshape(self.X.shape)
#         print("FLAT: d_input_shape = ", d_input.shape)

        return d_input
#         raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
