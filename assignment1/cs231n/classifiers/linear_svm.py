import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # gradient w.r.t not correct classifier
        dW[j,:] += X[:,i].T
        # gradient w.r.t correct classifier
        dW[y[i],:] -= X[:,i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  # Average gradients as well
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
    
  # Add regularization to the gradient
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(W, X) # (10, 49000)
  # (10, 49000)
  correct_scores = np.ones(scores.shape) * scores[y, np.arange(0, scores.shape[1])]
  margin = scores - correct_scores + 1
  margin[margin < 0] = 0

  # Set correct scores to 0
  margin[y, np.arange(0, scores.shape[1])] = 0
  loss = np.sum(margin)

  # Average over number of training examples
  loss /= scores.shape[1] # num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(np.square(W))
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  grad = np.zeros(scores.shape)
  margin = scores - correct_scores + 1
  margin[margin < 0] = 0
  margin[margin > 0] = 1
  margin[y, np.arange(0, margin.shape[1])] = 0
  margin[y, np.arange(0, margin.shape[1])] = -1 * np.sum(margin, axis=0)
  dW = np.dot(margin, X.T) # (10, 49000) * (49000, 3073)
  dW /= X.shape[1] # num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
