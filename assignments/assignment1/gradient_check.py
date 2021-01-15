import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
#     print("Check_phase_1")
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
#     print(orig_x)
    fx, analytic_grad = f(x)
#     print("Analytic = " + str(analytic_grad))
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

#     print("Check_phase_2")

    
    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) # multi_index causes a multi-index, or a tuple of indices with one per iteration dimension, to be tracked, 
    # readwrite indicates the operand will be read from and written to
    while not it.finished: # Whether the iteration over the operands is finished or not.
        ix = it.multi_index # When the multi_index flag was used, this property provides access to the index. 
#         print('ix' + str(ix))
        analytic_grad_at_ix = analytic_grad[ix]
        x_upper = orig_x.copy()
#         print("x_upper = " + str(x_upper))
#         print("x_upper[:,:]" + str(x_upper[:,:]))
#         print("x_upper[ix]" + str(x_upper[ix]))
        x_upper[ix] += delta
        x_lower = orig_x.copy()
        x_lower[ix] -= delta
#         print("Upper bound")
        fx_upper, _ = f(x_upper)
#         print("fx_upper = ", fx_upper)
#         print("Lower bound")
        fx_lower, _ = f(x_lower)
#         print("fx_lower = ", fx_lower)
        numeric_grad_at_ix = (fx_upper - fx_lower)/(2*delta)
#         print(f"Numeric on ix {ix} = {numeric_grad_at_ix}!")
        
#         print("Check_phase_3")
        
        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol).all():
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False
        
        it.iternext()

    print("Gradient check passed!")
    return True

        

        
