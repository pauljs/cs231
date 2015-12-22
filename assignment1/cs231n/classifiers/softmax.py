import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  losses = []
  for i in xrange(X.shape[1]):
        scores = np.dot(W, X[:, i])
        scores -= np.max(scores)
        loss -= scores[y[i]]
        loss += np.log(np.sum(np.exp(scores)))
        
        sum_exp = 0.0
        for score in scores:
            sum_exp += np.exp(score)
        
        for j in xrange(W.shape[0]):
            dW[j, :] += 1.0 / sum_exp * np.exp(scores[j]) * X[:, i]
            if j == y[i]:
                dW[j, :] -= X[:, i]

  # Average loss and dW over num_train
  num_train = X.shape[1]
  loss /= num_train
  dW /= num_train
  # Add regularization
  loss += 0.5 * reg * np.sum(np.square(W))
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
  num_train = X.shape[1]
  scores = np.dot(W, X) # (10, 49000)
  scores -= np.max(scores, axis=0) #(10, 49000) - (49000,)
  loss -= np.sum(scores[y, np.arange(0, num_train)])
  loss += np.sum(np.log(np.sum(np.exp(scores), axis=0)))
  
  sum_exp_scores = np.sum(np.exp(scores), axis=0) # (49000,)
  sum_exp_scores = 1.0 / (sum_exp_scores + 1e-8) # (49000,)
  dW = np.exp(scores) * sum_exp_scores #(10,49000)
  dW = np.dot(dW, X.T) #(10, 373)
  y_mat = np.zeros(shape=(W.shape[0], num_train)) # (10, 49000)
  y_mat[y, np.arange(0, num_train)] = 1 # (10, 49000)
  dW -= np.dot(y_mat, X.T) # (10, 49000) * (49000, 373)
    
  loss /= num_train
  loss += 0.5 * reg * np.sum(np.square(W))

  
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
