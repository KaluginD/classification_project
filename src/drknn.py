# https://github.com/meowoodie/Dr.k-NN/

import arrow
import torch
from torch import nn
import cvxpy as cp
import numpy as np
import torch.optim as optim
from cvxpylayers.torch import CvxpyLayer


def pairwise_dist(X, Y):
    """
    calculate pairwise l2 distance between X and Y
    """
    X_norm = (X**2).sum(dim=1).view(-1, 1)  # [n_xsample, 1]
    Y_t = torch.transpose(Y, 0, 1)  # [n_feature, n_ysample]
    Y_norm = (Y**2).sum(dim=1).view(1, -1)  # [1, n_ysample]
    dist = X_norm + Y_norm - 2.0 * torch.mm(X, Y_t)  # [n_xsample, n_ysample]
    return dist


def sortedY2Q(Y):
    """
    transform the sorted input label into the empirical distribution matrix Q, where
        Q_k^l = 1 / n_k, for n_{k-1} \le l \le n_{k+1}
              = 0, otherwise

    input
    - Y: [batch_size, n_sample]
    output
    - Q: [batch_size, n_class, n_sample]
    """
    batch_size, n_sample = Y.shape
    # NOTE:
    # it is necessary to require that the number of data points of each class in a single batch
    # is no less than 1 here.
    classes = torch.unique(Y)
    n_class = classes.shape[0]
    # N records the number of data points of each class in each batch [batch_size, n_class]
    N = [torch.unique(y, return_counts=True)[1] for y in Y.split(split_size=1)]
    N = torch.stack(N, dim=0)
    # construct an empty Q matrix with zero entries
    Q = torch.zeros(batch_size, n_class, n_sample)
    for batch_idx in range(batch_size):
        for class_idx in range(n_class):
            _from = N[batch_idx, :class_idx].sum()
            _to = N[batch_idx, : class_idx + 1].sum()
            n_k = N[batch_idx, class_idx].float()
            Q[batch_idx, class_idx, _from:_to] = 1 / n_k
    return Q


def tvloss(p_hat):
    """TV loss"""
    # p_max, _ = torch.max(p_hat, dim=1) # [batch_size, n_sample]
    # return p_max.sum(dim=1).mean()     # scalar
    p_min, _ = torch.min(p_hat, dim=1)  # [batch_size, n_sample]
    return p_min.sum(dim=1).mean()  # scalar


def evaluate_p_hat(H, Q, theta):
    n_class, n_sample, n_feature = theta.shape[1], H.shape[1], H.shape[2]
    rbstclf = RobustClassifierLayer(n_class, n_sample, n_feature)
    return rbstclf(H, Q, theta).data


def knn_regressor(H_test, H_train, p_hat_train, K=5):
    """
    k-Nearest Neighbor Regressor
    Given the train embedding and its corresponding optimal marginal distribution for each class,
    the function produce the prediction of p_hat for testing dataset given the test embedding based
    on the k-Nearest Neighbor rule.
    input
    - H_test:      [n_test_sample, n_feature]
    - H_train:     [n_train_sample, n_feature]
    - p_hat_train: [n_class, n_train_sample]
    output
    - p_hat_test:  [n_class, n_test_sample]
    """
    # find the indices of k-nearest neighbor in trainset
    dist = pairwise_dist(H_test, H_train)
    dist *= -1
    _, knb = torch.topk(dist, K, dim=1)  # [n_test_sample, K]
    # calculate the class marginal probability (p_hat) for each test sample
    p_hat_test = torch.stack(
        [p_hat_train[:, neighbors].mean(dim=1) for neighbors in knb], dim=0
    ).t()  # [n_class, n_test_sample]
    return p_hat_test


class RobustClassifierLayer(torch.nn.Module):
    """
    A Robust Classifier Layer via CvxpyLayer
    """

    def __init__(self, n_class, n_sample, n_feature):
        """
        Args:
        - n_class:  number of classes
        - n_sample: total number of samples in a single batch (including all classes)
        """
        super(RobustClassifierLayer, self).__init__()
        self.n_class, self.n_sample, self.n_feature = n_class, n_sample, n_feature
        self.cvxpylayer = self._cvxpylayer(n_class, n_sample)

    def forward(self, X_tch, Q_tch, theta_tch):
        """
        customized forward function.
        X_tch is a single batch of the input data and Q_tch is the empirical distribution obtained from
        the labels of this batch.
        input:
        - X_tch: [batch_size, n_sample, n_feature]
        - Q_tch: [batch_size, n_class, n_sample]
        - theta_tch: [batch_size, n_class]
        output:
        - p_hat: [batch_size, n_class, n_sample]
        """
        C_tch = self._wasserstein_distance(X_tch)  # [batch_size, n_sample, n_sample]
        gamma_hat = self.cvxpylayer(
            theta_tch, Q_tch, C_tch
        )  # (n_class [batch_size, n_sample, n_sample])
        gamma_hat = torch.stack(
            gamma_hat, dim=1
        )  # [batch_size, n_class, n_sample, n_sample]
        p_hat = gamma_hat.sum(dim=2)  # [batch_size, n_class, n_sample]
        return p_hat

    @staticmethod
    def _wasserstein_distance(X):
        """
        the wasserstein distance for the input data via calculating the pairwise norm of two aribtrary
        data points in the single batch of the input data, denoted as C here.
        see reference below for pairwise distance calculation in torch:
        https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2

        input
        - X: [batch_size, n_sample, n_feature]
        output
        - C_tch: [batch_size, n_sample, n_sample]
        """
        C_tch = []
        for x in X.split(split_size=1):
            x = torch.squeeze(x, dim=0)  # [n_sample, n_feature]
            x_norm = (x**2).sum(dim=1).view(-1, 1)  # [n_sample, 1]
            y_t = torch.transpose(x, 0, 1)  # [n_feature, n_sample]
            y_norm = x_norm.view(1, -1)  # [1, n_sample]
            dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)  # [n_sample, n_sample]
            # Ensure diagonal is zero if x=y
            dist = dist - torch.diag(dist)  # [n_sample, n_sample]
            dist = torch.clamp(dist, min=0.0, max=np.inf)  # [n_sample, n_sample]
            C_tch.append(dist)
        C_tch = torch.stack(C_tch, dim=0)  # [batch_size, n_sample, n_sample]
        return C_tch

    @staticmethod
    def _cvxpylayer(n_class, n_sample):
        """
        construct a cvxpylayer that solves a robust classification problem
        see reference below for the binary case:
        http://papers.nips.cc/paper/8015-robust-hypothesis-testing-using-wasserstein-uncertainty-sets
        """
        # NOTE:
        # cvxpy currently doesn't support N-dim variables, see discussion and solution below:
        # * how to build N-dim variables?
        #   https://github.com/cvxgrp/cvxpy/issues/198
        # * how to stack variables?
        #   https://stackoverflow.com/questions/45212926/how-to-stack-variables-together-in-cvxpy

        # Variables
        # - gamma_k: the joint probability distribution on Omega^2 with marginal distribution Q_k and p_k
        gamma = [cp.Variable((n_sample, n_sample)) for k in range(n_class)]
        # - p_k: the marginal distribution of class k [n_class, n_sample]
        p = [cp.sum(gamma[k], axis=0) for k in range(n_class)]
        p = cp.vstack(p)

        # Parameters (indirectly from input data)
        # - theta: the threshold of wasserstein distance for each class
        theta = cp.Parameter(n_class)
        # - Q: the empirical distribution of class k obtained from the input label
        Q = cp.Parameter((n_class, n_sample))
        # - C: the pairwise distance between data points
        C = cp.Parameter((n_sample, n_sample))

        # Constraints
        cons = [g >= 0.0 for g in gamma]
        for k in range(n_class):
            cons += [cp.sum(cp.multiply(gamma[k], C)) <= theta[k]]
            for l in range(n_sample):
                cons += [cp.sum(gamma[k], axis=1)[l] == Q[k, l]]

        # Problem setup
        # tv loss
        obj = cp.Maximize(cp.sum(cp.min(p, axis=0)))
        # cross entropy loss
        # obj   = cp.Minimize(cp.sum(- cp.sum(p * cp.log(p), axis=0)))
        prob = cp.Problem(obj, cons)
        assert prob.is_dpp()

        # return cvxpylayer with shape (n_class [batch_size, n_sample, n_sample])
        # stack operation ('torch.stack(gamma_hat, axis=1)') is needed for the output of this layer
        # to convert the output tensor into a normal shape, i.e., [batch_size, n_class, n_sample, n_sample]
        return CvxpyLayer(prob, parameters=[theta, Q, C], variables=gamma)


class RobustTextClassifier(nn.Module):
    def __init__(self, n_class, n_sample, max_theta=0.1, n_iter=100, lr=1e-2, K=5):
        super(RobustTextClassifier, self).__init__()
        self.n_class = n_class
        self.n_sample = n_sample
        self.max_theta = max_theta
        self.n_iter = n_iter
        self.lr = lr
        self.K = K

        self.proj = nn.Linear(384, 192)
        self.theta = nn.Parameter(torch.ones(self.n_class).float() * self.max_theta)
        self.theta.requires_grad = True
        # self.theta     = torch.ones(self.n_class) * self.max_theta
        self.rbstclf = RobustClassifierLayer(n_class, n_sample, 192)

    def forward(self, X, Q):
        batch_size = X.shape[0]
        n_sample = X.shape[1]

        # CNN layer
        # NOTE: merge the batch_size dimension and n_sample dimension
        X = X.reshape(
            batch_size * n_sample, X.shape[2]
        )  # [batch_size*n_sample, in_channel, n_pixel, n_pixel]
        Z = self.proj(X)  # [batch_size*n_sample, n_feature]
        # NOTE: reshape back to batch_size and n_sample
        Z = Z.reshape(
            batch_size, n_sample, Z.shape[-1]
        )  # [batch_size, n_sample, n_feature]

        # robust classifier layer
        # theta = torch.ones(batch_size, self.n_class, requires_grad=True) * self.max_theta
        theta = self.theta.unsqueeze(0).repeat([batch_size, 1])  # [batch_size, n_class]
        p_hat = self.rbstclf(Z, Q, theta)  # [batch_size, n_class, n_sample]
        return p_hat

    def fit(self, X, y):
        self.X_train = torch.from_numpy(X).float()
        self.y_train = torch.from_numpy(y).reshape((1, -1))
        optimizer = optim.Adadelta(self.parameters(), lr=self.lr)

        for i in range(self.n_iter):
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            pos_idx, neg_idx = np.random.choice(pos, 2), np.random.choice(neg, 2)

            X_batch = torch.from_numpy(X[np.concatenate((pos_idx, neg_idx))]).float()
            y_batch = torch.from_numpy(
                np.concatenate((np.ones(pos_idx.shape), np.zeros(neg_idx.shape)))
            )

            X_batch = X_batch.reshape(2, 2, 384).transpose(1, 0)
            y_batch = y_batch.reshape(2, 2).T

            Q = sortedY2Q(y_batch)
            optimizer.zero_grad()
            p_hat = self.forward(X_batch, Q)
            loss = tvloss(p_hat)
            loss.backward()
            optimizer.step()

        self.Q = sortedY2Q(self.y_train)
        self.H_train = self.proj(self.X_train)

    def predict(self, X_test):
        H_test = self.proj(torch.from_numpy(X_test).float())
        theta = self.theta.data.unsqueeze(0)
        p_hat = evaluate_p_hat(self.H_train.unsqueeze(0), self.Q, theta).squeeze(0)
        p_hat_knn = knn_regressor(H_test, self.H_train, p_hat, self.K)
        knn_pred = p_hat_knn.argmax(dim=0)
        return knn_pred.numpy()
