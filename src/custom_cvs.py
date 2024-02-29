from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable

import numpy as np


class VariableTimeSeriesSplit(TimeSeriesSplit):
    """Variable Time Series cross-validator

    Provides train/test indices to split time series data samples
    that are observed at variable time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`TimeSeriesSplit`.
    Instead of being fixed time intervals, it allows to variable time
    intervals on each split.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Read more in the :ref:`User Guide <time_series_split>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.

    max_train_size : int, list or ndarray, default=None
        Maximum size for a single training set, globally or per split.

    test_size : int, list or ndarray, default=None
        Used to limit the size of the test set globally or per split. Defaults
        to ``n_samples // (n_splits + 1)``, which is the maximum allowed value
        with ``gap=0``.

    gap : int, list or ndarray, default=0
        Number of samples to exclude from the end of each train set before
        the test set, globally or per split.

    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)``
    + ``n_samples % (n_splits + 1)`` in the ``i`` th split,
    with a test set of size ``n_samples//(n_splits + 1)`` by default,
    where ``n_samples`` is the number of samples.
    """

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1

        max_train_sizes = self._norm_param(
            self.max_train_size, np.repeat(None, n_splits)
        )
        test_sizes = self._norm_param(
            self.test_size, np.repeat(n_samples // n_folds, n_splits)
        )
        gaps = self._norm_param(self.gap)

        if len(max_train_sizes) != n_splits:
            raise ValueError(f"The size of max_train_size does not match n_splits.")
        if len(test_sizes) != n_splits:
            raise ValueError(f"The size of test_size does not match n_splits.")
        if len(gaps) != n_splits:
            raise ValueError(f"The size of gap does not match n_splits.")

        cond = n_samples - gaps - (test_sizes * n_splits) <= 0
        if np.any(cond):
            raise ValueError(
                f"Too many splits={n_splits} for number of samples"
                f"def={n_samples} with test_size={test_sizes[cond]} and gap={gaps[cond]}."
            )
        if n_folds > n_samples:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of samples={n_samples}."
            )

        indices = np.arange(n_samples)
        test_starts = n_samples - test_sizes[::-1].cumsum()[::-1]
        train_ends = test_starts - gaps

        for i in range(n_splits):
            if max_train_sizes[i] and max_train_sizes[i] < train_ends[i]:
                yield (
                    indices[train_ends[i] - max_train_sizes[i] : train_ends[i]],
                    indices[test_starts[i] : test_starts[i] + test_sizes[i]],
                )
            else:
                yield (
                    indices[: train_ends[i]],
                    indices[test_starts[i] : test_starts[i] + test_sizes[i]],
                )

    def _norm_param(self, param, none_case=None):
        if param is None:
            return none_case
        elif type(param) == np.ndarray:
            return param
        elif type(param) == list:
            return np.array(param)
        else:
            return np.repeat(param, self.n_splits)
