from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: 执行向前传播，计算输入数据的每个类的score。
        # 将结果存储在scores变量中，该变量应该是一个(N, C)维的数组。
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        fc1_h = np.dot(X,W1)+b1                                   # (N,H) 第一层全连接，使用了广播机制
        fc1_a = np.maximum(fc1_h, 0)                                # (N,H)  ReLU激活
        fc2_h = np.dot(fc1_a,W2)+b2                                # (N,C) 第二层全连接
        scores = fc2_h                                          # (N,C)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: 完成向前传播，计算损失。
        # 这应该包括数据损失和W1和W2的L2正则化项。
        # 将结果存储在变量loss中，它应该是一个标量。
        # 使用Softmax损失函数。
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        scores = scores - np.max(X,axis=1,keepdims=True)                             # (N,C) 处理稳定性问题
        softmax = np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True)              # (N,C) 后面还要用，所以这里一块计算了
                                                                     # 这里暂时没有减去最大值
        softmax_i = softmax[np.arange(N),y]                                      #  (N)
        cross_entropy_loss = -np.log(softmax_i)                                   # (N) 也可以用全1向量乘积代替sum运算
        loss = 1/N * cross_entropy_loss.sum() + reg * (np.square(W1).sum()+np.square(W2).sum())   # (1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: 计算反向传播，计算权重和偏置值的梯度, 将结果存储在grads字典中。
        # 例如，grads['W1']存储W1的梯度，并且和W1是相同大小的矩阵。
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # 下面的过程为了体现运算过程有些地方并没有进行简化
        
        # 对fc2_h求导
        # Ref: https://zhuanlan.zhihu.com/p/25723112 这里改为向量化运算
        tmp_loss_grad_fc2_h = softmax                                           # (N,C)  注意还有正则化项的loss
        tmp_loss_grad_fc2_h[np.arange(N),y] -= 1                                   # (N,C) 具体过程参见上述链接
        loss_grad_fc2_h = 1/N * tmp_loss_grad_fc2_h                                 # (N,C) 具体过程参见上述链接
        
        # 对W2求导
        fc2_h_grad_W2 = fc1_a.T                                              # (H,N)
        reg_grad_W2 = 2 * reg * W2                                            # (H,C)
        grads['W2'] = fc2_h_grad_W2.dot(loss_grad_fc2_h) + reg_grad_W2                    # (H,C)
        
        # 对b2求导
        fc2_h_grad_b2 = np.identity(N)                                          # (N,N) 其实为np.identity(N).T             
        grads['b2'] = np.sum(fc2_h_grad_b2.dot(loss_grad_fc2_h),axis=0)                    # (C) 注意这里的广播机制
        
        # 对fc1_h求导
        loss_grad_fc1_a = loss_grad_fc2_h.dot(W2.T)                                 # (N,H)
        loss_grad_fc1_h = loss_grad_fc1_a.copy()                                  # (N,H)
        loss_grad_fc1_h[fc1_h < 0] = 0                                         # (N,H)
        
        # 对W1求导
        fc1_h_grad_W1 = X.T                                                 # (D,N)
        reg_grad_W1 = 2 * reg * W1                                            # (D,H)
        grads['W1'] = fc1_h_grad_W1.dot(loss_grad_fc1_h) + reg_grad_W1                    # (D,H)
        
        # 对b1求导
        fc1_h_grad_b1 = np.identity(N)                                          # (N,N)
        grads['b1'] = np.sum(fc1_h_grad_b1.dot(loss_grad_fc1_h),axis=0)                    # (H)         
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: 创建一个随机的数据和标签的mini-batch，存储在X_batch和y_batch中。
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            mask = np.random.choice(a = num_train, size=batch_size,replace=True)  
            X_batch = X[mask]
            y_batch = y[mask]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: 使用grads字典中的梯度来更新网络参数(参数存储在字典self.params中)
            # 使用随机梯度下降法。
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            for p in self.params:
                self.params[p] -= learning_rate * grads[p]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        y_pred = np.argmax(self.loss(X),axis=1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
