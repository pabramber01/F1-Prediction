import numpy as np


def ranker_predict(ranker, group, multiclass):
    """Predict over a rank model the position in variable interval or if it is
    podium.

    Parameters
    ----------
    ranker : ranker model
        Ranker model with which is going to predict group.

    group : array-like of shape (group_samples, n_attributes)
        Group instances to predict.

    multiclass : bool
        Indicator if it is going to predict variable interval or podium.

    Returns
    -------
    ls : array-like of shape (group_samples,)
        Ranking predicted. Notice if it is podium, it is a list of zeros where
        there are three ones which are the three first positions.
    """
    if 0 in group.columns:
        pred = ranker.predict(group.to_numpy())
    else:
        pred = ranker.predict(group.loc[:, ~group.columns.isin(["qid"])])

    idx = np.argsort(pred) if multiclass else np.argsort(pred)[::-1]
    ls = np.zeros_like(pred)
    ls[idx] = np.arange(1, len(pred) + 1)

    if not multiclass:
        ls[ls <= 3] = 1
        ls[ls > 3] = 0

    return ls
