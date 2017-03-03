"""
Logistic Regression
"""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Fabian Pedregosa <f@bianp.net>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Manoj Kumar <manojkumarsivaraj334@gmail.com>
#         Lars Buitinck
#         Simon Wu <s8wu@uwaterloo.ca>

import numbers
import warnings

import numpy as np
from scipy import optimize, sparse

from .base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
from .sag import sag_solver
from ..preprocessing import LabelEncoder, LabelBinarizer
from ..svm.base import _fit_liblinear
from ..utils import check_array, check_consistent_length, compute_class_weight
from ..utils import check_random_state
from ..utils.extmath import (logsumexp, log_logistic, safe_sparse_dot,
                             softmax, squared_norm)
from ..utils.extmath import row_norms
from ..utils.optimize import newton_cg
from ..utils.validation import check_X_y
from ..exceptions import NotFittedError
from ..utils.fixes import expit
from ..utils.multiclass import check_classification_targets
from ..externals.joblib import Parallel, delayed
from ..model_selection import check_cv
from ..externals import six
from ..metrics import SCORERS


def _intercept_dot(w, X, y):
    c = np.zeros(w.shape[1])
    if w.shape[0] == X.shape[1] + 1:
        c = w[-1, :]
        w = w[:-1, :]
    z = safe_sparse_dot(X, w) + c
    yz = y * z
    return w, c, yz


def _logistic_loss(w, X, y, alpha, sample_weight=None):
    w, c, yz = _intercept_dot(w, X, y)

    if sample_weight is None:
        sample_weight = np.ones((y.shape[0], 1))
    out = -np.sum(sample_weight * log_logistic(yz)) + .5 * alpha * squared_norm(w)
    return out.sum()


def _logistic_loss_and_grad(w, X, y, alpha, sample_weight=None):
    n_samples, n_features = X.shape
    n_samples, n_labels = y.shape
    fit_intercept = (w.size == (n_features + 1)*n_labels)
    w = w.reshape((n_features+bool(fit_intercept), n_labels))
    grad = np.empty_like(w)
    w, c, yz = _intercept_dot(w, X, y)

    if sample_weight is None:
        sample_weight = np.ones((y.shape[0], 1))

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(sample_weight * log_logistic(yz)) + .5 * alpha * squared_norm(w)

    z = expit(yz)
    z0 = sample_weight * (z - 1) * y
    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w
    # Case where we fit the intercept.
    if fit_intercept:
        grad[-1] = z0.sum(axis=0)
    return out, grad.ravel()




def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=1,
                             solver='lbfgs', coef=None,
                             class_weight=None, dual=False, penalty='l2',
                             intercept_scaling=1., multi_class='ovr',
                             random_state=None, check_input=True,
                             max_squared_sum=None, sample_weight=None):
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    # Preprocessing.
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        y = check_array(X, accept_sparse='csr', dtype=np.float64)
        check_consistent_length(X, y)
    _, n_features = X.shape
    _, n_labels = y.shape
    classes = np.unique(y)
    random_state = check_random_state(random_state)

    sample_weight = np.ones((X.shape[0], 1))

    w0 = np.zeros((n_features + int(fit_intercept), n_labels),
                  order='F')

    target = 2*y - 1

    if multi_class == 'multi-label':
        # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.
        if solver in ['lbfgs', 'newton-cg']:
            w0 = w0.ravel()
        if solver == 'lbfgs':
            func = lambda x, *args: _logistic_loss_and_grad(x, *args)
        warm_start_sag = {'coef': w0.T}

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        if solver == 'lbfgs':
            try:
                w0, loss, info = optimize.fmin_l_bfgs_b(
                    func, w0, fprime=None,
                    args=(X, target, 1. / C, sample_weight),
                    iprint=(verbose > 0) - 1, pgtol=tol, maxiter=max_iter)
            except TypeError:
                # old scipy doesn't have maxiter
                w0, loss, info = optimize.fmin_l_bfgs_b(
                    func, w0, fprime=None,
                    args=(X, target, 1. / C, sample_weight),
                    iprint=(verbose > 0) - 1, pgtol=tol)
            if info["warnflag"] == 1 and verbose > 0:
                warnings.warn("lbfgs failed to converge. Increase the number "
                              "of iterations.")
            try:
                n_iter_i = info['nit'] - 1
            except:
                n_iter_i = info['funcalls'] - 1
        else:
            raise ValueError("solver must be one of {'liblinear', 'lbfgs', "
                             "'newton-cg', 'sag'}, got '%s' instead" % solver)

        coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return coefs, np.array(Cs), n_iter


class MultiLogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100,
                 multi_class='multi-label', verbose=0, warm_start=False, n_jobs=1):

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.

        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64, multi_output=True,
                         order="C")
        check_classification_targets(y)
        n_samples, n_labels = y.shape
        n_samples, n_features = X.shape

        max_squared_sum = None
        classes = np.unique(y)
        self.classes_ = classes
        if len(classes) == 2:
            classes = classes[1:]

        self.coef_ = list()
        self.intercept_ = np.zeros(n_labels)
        
        
        if self.warm_start:
            warm_start_coef = getattr(self, 'coef_', None)
        else:
            warm_start_coef = None

        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(warm_start_coef,
                                        self.intercept_[np.newaxis,:],
                                        axis=0)

        if warm_start_coef is None:
            warm_start_coef = [None] * n_labels

        path_func = delayed(logistic_regression_path)
        
        backend = 'multiprocessing'
        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               backend=backend)(
            path_func(X, y, pos_class=class_, Cs=[self.C],
                      fit_intercept=self.fit_intercept, tol=self.tol,
                      verbose=self.verbose, solver=self.solver,
                      multi_class=self.multi_class, max_iter=self.max_iter,
                      class_weight=self.class_weight, check_input=False,
                      random_state=self.random_state, coef=warm_start_coef_,
                      max_squared_sum=max_squared_sum,
                      sample_weight=sample_weight)
            for (class_, warm_start_coef_) in zip(classes, warm_start_coef))
        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        self.coef_ = fold_coefs_[0][0]
        self.coef_ = self.coef_.reshape(n_features +
                                        int(self.fit_intercept), n_labels)

        if self.fit_intercept:
            self.intercept_ = self.coef_[-1,:]
            self.coef_ = self.coef_[:-1, :]

        return self
    
    def decision_function(self, X):
        if not hasattr(self, "coef_"):
            raise NotFittedError("Call fit before prediction")

        return safe_sparse_dot(X, self.coef_)+self.intercept_

    def predict_proba(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        return prob

    def predict(self, X):
        scores = self.decision_function(X)
        indices = (scores>0).astype(np.int)
        return self.classes_[indices]
