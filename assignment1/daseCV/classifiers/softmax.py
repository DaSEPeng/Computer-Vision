from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # TODO: 使用显式循环计算softmax损失及其梯度。
    # 将损失和梯度分别保存在loss和dW中。
    # 如果你不小心，很容易遇到数值不稳定的情况。 
    # 不要忘了正则化！                                                           
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N = X.shape[0]
    D,C = W.shape
    
    scores = np.zeros((N,C))
    softmax = np.zeros((N,C))
    cross_entropy_loss = 0.0
    W_square_sum = 0.0
    
    # 计算交叉熵损失
    for i in range(N):
        sum_exp_i = 0.0
        for j in range(C):
            scores[i,j] = X[i,:].dot(W[:,j])                              # 计算得分
            softmax[i,j] = np.exp(scores[i,j])                             # 暂时的softmax
            sum_exp_i += softmax[i,j]
        scores[i,:] -= np.max(scores[i,:])                                # 处理稳定性问题       
        softmax[i,:] /= sum_exp_i                                      # 真正的softmax   
        cross_entropy_loss -= np.log(softmax[i,y[i]])
    
    # 计算正则化项
    for i in range(D):
        for j in range(C):
            W_square_sum += W[i,j]*W[i,j]
            
    # 计算最终的损失
    loss = 1/N * cross_entropy_loss + reg * W_square_sum                         
    
    # 计算dsoftmax
    tmp_dsoftmax = softmax                                
    for i in range(N):
        for j in range(C):
            if j == y[i]:
                tmp_dsoftmax[i,j] -= 1
    dsoftmax = 1/N * tmp_dsoftmax      
    
    # 计算dW
    for i in range(D):
        for j in range(C):
            dW[i,j] = X[:,i].dot(dsoftmax[:,j]) + 2 * reg * W[i,j]                                      

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # TODO: 不使用显式循环计算softmax损失及其梯度。
    # 将损失和梯度分别保存在loss和dW中。
    # 如果你不小心，很容易遇到数值不稳定的情况。 
    # 不要忘了正则化！
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N = X.shape[0]

    scores = X.dot(W)                                                 # (N,C)                                 
    scores = scores - np.max(X,axis=1,keepdims= True)                           # (N,C) 处理稳定性问题
    
    softmax = np.exp(scores)/np.sum(np.exp(scores),axis=1,keepdims=True)              # (N,C)  
    
    softmax_i = softmax[np.arange(N),y]                                     # (N)
    cross_entropy_loss = -np.log(softmax_i)                                   # (N) 
    
    loss = 1/N * cross_entropy_loss.sum() + reg * np.square(W).sum()                  # (1)
    
    tmp_dsoftmax = softmax                                                 # (N,C)
    tmp_dsoftmax[np.arange(N),y] -= 1                                         # (N,C)
    dsoftmax = 1/N * tmp_dsoftmax                                           # (N,C)
    dW = X.T.dot(dsoftmax) + 2*reg*W                                         # (D,C)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
