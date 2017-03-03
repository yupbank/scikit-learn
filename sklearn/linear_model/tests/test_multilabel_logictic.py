import numpy as np
import scipy.sparse as sp
from scipy import linalg, optimize, sparse

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_warns
from sklearn.utils.testing import raises
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_raise_message
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import compute_class_weight
from sklearn.utils.fixes import sp_version

from ..logistic import (
    _intercept_dot as _single_intercept_dot,
    _logistic_loss as _single_logistic_loss,
    _logistic_loss_and_grad as _single_logistic_loss_and_grad,
    _logistic_grad_hess as _single_logistic_grad_hess,
    LogisticRegression
)
from sklearn.linear_model.multi_label_logistic import (
    _intercept_dot,
    _logistic_loss,
    _logistic_loss_and_grad,
    MultiLogisticRegression
    )
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

X = [[-1, 0], [0, 1], [1, 1]]
X_sp = sp.csr_matrix(X)
Y1 = [0, 1, 1]
Y2 = [2, 1, 0]
iris = load_iris()

def test_intercept_dot_multi_label():
    n_samples, n_features = 10, 5
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               random_state=0)

    # Fit intercept case.
    alpha = 1.
    w = np.ones(n_features + 1)
    multi_w = np.ones((n_features+1, 2))
    multi_y = np.hstack([y[:,np.newaxis], y[:, np.newaxis]])
    single_w, single_c, single_yz = _single_intercept_dot(w, X, y)
    new_w, new_c, new_yz = _intercept_dot(multi_w, X, multi_y)
    expected_w = np.hstack([single_w[:,np.newaxis], single_w[:, np.newaxis]])
    expected_c = np.array([single_c, single_c])
    expected_yz = np.hstack([single_yz[:,np.newaxis], single_yz[:, np.newaxis]])
    assert_array_equal(expected_w, new_w)
    assert_array_equal(expected_c, new_c)
    assert_almost_equal(expected_yz, new_yz)


def test_logistic_loss_multi_label():
    n_samples, n_features = 10, 5
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               random_state=0)

    # Fit intercept case.
    alpha = 1.
    w = np.ones(n_features + 1)
    multi_w = np.ones((n_features+1, 2))
    multi_y = np.hstack([y[:,np.newaxis], y[:, np.newaxis]])
    
    sinlge_loss = _single_logistic_loss(w, X, y, alpha)
    expected_loss = np.array([sinlge_loss, sinlge_loss]) 

    multi_loss = _logistic_loss(multi_w, X, multi_y, alpha)
    assert_almost_equal(expected_loss.sum(), multi_loss)


def test_logitc_loss_grad_multi_label():
    n_samples, n_features = 10, 5
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               random_state=0)

    # Fit intercept case.
    alpha = 1.
    w = np.ones(n_features + 1)
    multi_w = np.ones((n_features+1, 2))
    multi_y = np.hstack([y[:,np.newaxis], y[:, np.newaxis]])

    single_loss, single_grad = _single_logistic_loss_and_grad(w, X, y, alpha)
    multi_loss, multi_grad = _logistic_loss_and_grad(multi_w, X, multi_y, alpha)

    expected_grad = np.hstack([single_grad[:,np.newaxis], single_grad[:, np.newaxis]])
    expected_loss = np.array([single_loss, single_loss]) 
    assert_almost_equal(expected_loss.sum(), multi_loss)
    assert_almost_equal(expected_grad.ravel(), multi_grad)


def test_multilogistic():
    n_samples, n_features = 10, 5
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               random_state=0)

    alpha = 1.
    multi_y = np.hstack([y[:,np.newaxis], y[:, np.newaxis]])
    clf = MultiLogisticRegression(fit_intercept=False)
    clf.fit(X, multi_y[:,1:])
    pred_y = clf.predict(X)
    assert_almost_equal(pred_y.ravel(), y.ravel())
    clf.fit(X, multi_y)
    pred_y = clf.predict(X)
    assert_almost_equal(pred_y.ravel(), multi_y.ravel())

