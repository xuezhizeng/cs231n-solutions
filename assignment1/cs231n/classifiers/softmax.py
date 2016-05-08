import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in xrange(num_train):
        
    scores = np.dot(X[i], W)
    correct_class_score = scores[y[i]]
    log_c = -np.max(scores)
    
    nominator = np.exp(correct_class_score + log_c)
    denominator = 0
    for j in xrange(num_classes):
        denominator += np.exp(scores[j] + log_c)
    loss += -np.log(nominator/denominator)
    
    for j in xrange(num_classes):
        p = np.exp(scores[j]+log_c)/denominator
        dW[:,j] += (p-1 if j == y[i] else p) * X[i, :]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_exp = X.shape[0]
    s = np.dot(X, W)
    log_c = -np.max(s)
    exp_s = np.exp(s+log_c)
    denominators = np.sum(exp_s, 1)

    nominators = exp_s[np.arange(num_exp), y]
    p = nominators/denominators
    loss = np.sum(-np.log(p))
    loss /= num_exp
    loss += 0.5 * reg * np.sum(W*W)
    
    p = exp_s/denominators.reshape(-1,1)
    p[np.arange(num_exp), y] -= 1
    dW = np.dot(X.T, p)
    dW /= num_exp
    dW += reg * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

