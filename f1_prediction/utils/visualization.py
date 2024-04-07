from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, make_scorer
from sklearn.inspection import permutation_importance

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from pygad import GA

from optuna import logging, create_study
from optuna.visualization.matplotlib import plot_optimization_history

logging.set_verbosity(logging.WARNING)

from scipy import stats

from utils.custom_scorers import balanced_accuracy_score
from utils.custom_cvs import VariableTimeSeriesSplit
from utils.custom_predicts import ranker_predict

import sys
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def model_test(
    models,
    X,
    y,
    cv,
    *,
    scorers=[make_scorer(balanced_accuracy_score), "neg_mean_absolute_error"],
    interval=None,
    rank=None,
    multiclass=None,
    labels=None,
):
    """Test the models.

    Compute cross validations, classification report and confusion matrix.

    Parameters
    ----------
    models : array-like
        Models to test.

    X : array-like of shape (n_samples, n_attributes)
        Train attributes.

    y : array-like of shape (n_samples,)
        Target attributes.

    cv : cv object
        cv object to pass to cross validation.

    scorers : array-like, default=[make_scorer(balanced_accuracy_score), "neg_mean_absolute_error"]
        Scorers to do the cross validation.

    interval : int, default=None
        Interval where y_pred is consider as hit.

    rank : bool, default=None
        Indicator if it is used to rank or not.

    multiclass : bool, default=None
        Indicator that it is used for rank variable position or podium.

    labels : array-like, default=None
        Labels for classification report and confusion matrix.
    """

    _, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(15, 6))

    for i, model in enumerate(models):

        print(f"{model}:")

        for scorer in scorers:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            if rank:
                print(f"CV with {scorer.__name__}:", scores.sum() / len(scores))
            else:
                print(f"CV with {scorer}:", scores.sum() / len(scores))

        splits = [
            (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])
            for (train, test) in cv.split(X)
        ]

        y_tests, y_preds = np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        for split in splits:
            X_train, X_test, y_train, y_test = split
            model.fit(X_train, y_train)
            y_tests = np.append(y_tests, y_test.values)
            if rank:
                y_preds = np.append(
                    y_preds,
                    X_test.groupby("qid")
                    .apply(lambda x: ranker_predict(model, x, multiclass))
                    .explode()
                    .reset_index(drop=True)
                    .to_numpy(dtype=int),
                )
            else:
                y_preds = np.append(y_preds, np.rint(model.predict(X_test)).astype(int))

        if interval is not None:
            y_preds = np.where(np.abs(y_tests - y_preds) <= interval, y_tests, y_preds)

        if labels is not None:
            y_tests, y_preds = labels[y_tests], labels[y_preds]

        print(classification_report(y_tests, y_preds, labels=labels))

        ax = axes[i] if len(models) > 1 else axes
        ConfusionMatrixDisplay.from_predictions(
            y_tests, y_preds, xticks_rotation="vertical", ax=ax, labels=labels
        )
        ax.title.set_text(type(model).__name__)

    plt.tight_layout()
    plt.grid(False)
    plt.show()


def model_selection(model, X, y, cv, scoring, *, rank=None):
    """Test the model with selected features.

    Compute sequential forward selector, permutation importance and
    genetic algorithm.

    Also plot boxplot from permutation importance and variations on
    sequential forward selector.

    Parameters
    ----------
    model : array-like
        Model to test.

    X : array-like of shape (n_samples, n_attributes)
        Train attributes.

    y : array-like of shape (n_samples,)
        Target attributes.

    cv : cv object
        cv object to compute on the three methods.

    scoring : scoring object
        Scorer to compute on the three methods.

    rank : bool, default=None
        Indicator if it is used to rank or not.
    """

    def fitness_func(ga_instance, individual, individual_idx):
        res = []

        if rank:
            individual[X.columns.get_loc("qid")] = 1

        idx = [i for i in range(len(individual)) if individual[i] == 1]

        attributes = X.iloc[:, idx]
        objective = y

        if not attributes.empty:
            res.extend(
                cross_val_score(
                    estimator=model,
                    X=attributes,
                    y=objective,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                )
            )

        avg_cross_val_score = sum(res) / len(res) if not attributes.empty else -10000
        return avg_cross_val_score

    # Permutation Importance
    model.fit(X, y)
    r = permutation_importance(
        model,
        X,
        y,
        n_jobs=-1,
        scoring=scoring,
        n_repeats=5,
    )

    # Sequential Forward Selector
    sfs = SFS(
        model,
        k_features=(1, len(X.columns)),
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
    )
    if rank:
        sfs.fit(X, y, qid=X.qid)
    else:
        sfs.fit(X, y)

    # Genetic Algorithms
    individual_size = X.shape[1]
    ga_instance = GA(
        num_generations=sys.maxsize,
        num_parents_mating=2,
        fitness_func=fitness_func,
        sol_per_pop=25,
        num_genes=individual_size,
        gene_type=int,
        init_range_low=0,
        init_range_high=2,
        parent_selection_type="rws",
        keep_parents=1,
        crossover_type="single_point",
        crossover_probability=0.3,
        mutation_type="random",
        mutation_probability=0.05,
        stop_criteria=["saturate_5"],
    )
    ga_instance.run()

    # PI results
    mean = r.importances_mean
    std = r.importances_std
    pos = lambda i: mean[i] - 2 * std[i] > 0
    feature_idx = sorted([i for i in mean.argsort()[::-1] if pos(i)])
    feature_names = sorted([X.columns[i] for i in mean.argsort()[::-1] if pos(i)])
    scores = cross_val_score(
        model, X[feature_names], y, cv=cv, scoring=scoring, n_jobs=-1
    )
    output = "PI: %.3f with %s == %s" % (
        scores.sum() / len(scores),
        tuple(feature_idx),
        tuple(feature_names),
    )
    print("\n".join(textwrap.wrap(output, 88, subsequent_indent="\t")))

    # SFS results
    output = "SFS: %.3f with %s == %s" % (
        sfs.k_score_,
        sfs.k_feature_idx_,
        sfs.k_feature_names_,
    )
    print("\n".join(textwrap.wrap(output, 88, subsequent_indent="\t")))

    # GA results
    solution, solution_fitness, _ = ga_instance.best_solution()
    feature_idx = np.where(solution == 1)[0]
    feature_names = X.columns[feature_idx]
    output = "GA: %.3f with %s == %s" % (
        solution_fitness,
        tuple(feature_idx),
        tuple(feature_names),
    )
    print("\n".join(textwrap.wrap(output, 88, subsequent_indent="\t")))

    # Plot results
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    dfp = pd.DataFrame(data=r.importances.T, columns=X.columns)
    meds = dfp.mean()
    meds.sort_values(ascending=True, inplace=True)
    dfp = dfp[meds.index]
    dfp.boxplot(vert=False, ax=ax1)
    ax1.title.set_text("Permutation Importance")
    ax1.set_xlabel("Importance")
    ax1.set_ylabel("Features")

    metric_dict = sfs.get_metric_dict()
    k_feat = sorted(metric_dict.keys())
    avg = [metric_dict[k]["avg_score"] for k in k_feat]
    upper = [metric_dict[k]["avg_score"] + metric_dict[k]["std_dev"] for k in k_feat]
    lower = [metric_dict[k]["avg_score"] - metric_dict[k]["std_dev"] for k in k_feat]
    ax2.plot(k_feat, avg, color="blue", marker="o")
    ax2.fill_between(k_feat, upper, lower, alpha=0.2, color="steelblue", lw=1)
    ax2.set_xticks(range(1, len(X.columns) + 1))
    ax2.title.set_text("Sequential Forward Selection (w. StdDev)")
    ax2.set_xlabel("Number of Features")
    ax2.set_ylabel("Performance")
    ax2.grid()

    plt.tight_layout()
    plt.show()


def model_tuning(params, model, X, y, cv, scoring):
    """Test the model with optimized hyperparameters.

    Compute optuna hyperparamenter optimization.

    Also plot performance per hyperparamenter combination.

    Parameters
    ----------
    params : array-like
        Parameters for optuna trial.

    model : array-like
        Model to test.

    X : array-like of shape (n_samples, n_attributes)
        Train attributes.

    y : array-like of shape (n_samples,)
        Target attributes.

    cv : cv object
        cv object to compute on the three methods.

    scoring : scoring object
        Scorer to compute on the three methods.
    """

    def objective(trial):
        params_trial = dict()
        for param in params:
            type = param[0]
            data = param[1]
            if type == "int":
                params_trial[data[0]] = trial.suggest_int(*data)
            elif type == "float":
                params_trial[data[0]] = trial.suggest_float(*data)
            elif type == "categorical":
                params_trial[data[0]] = trial.suggest_categorical(*data)
            elif type == "mlp":
                layers = []
                n_layers = trial.suggest_int("n_layers", 1, data[1])
                for i in range(n_layers):
                    layers.append(trial.suggest_int(f"n_units_{i}", 1, data[2]))
                params_trial[data[0]] = tuple(layers)

        score = cross_val_score(
            model(**params_trial),
            X,
            y,
            n_jobs=-1,
            cv=cv,
            scoring=scoring,
        )

        return sum(score) / len(score)

    study = create_study(direction="maximize")
    study.optimize(objective, n_trials=150, n_jobs=-1)

    output = "%s: %.3f with %s" % (
        model.__name__,
        study.best_trial.values[0],
        study.best_trial.params,
    )
    print("\n".join(textwrap.wrap(output, 88, subsequent_indent="\t")))
    plot_optimization_history(study, target_name="Performance")
    plt.show()


def model_pgrmlly_outlier_treatment(model, df, X, y, scorer, th):
    """Programmatically treats outliers.

    Plot performance per threshold.

    Parameters
    ----------
    params : array-like
        Parameters for optuna trial.

    model : array-like
        Model to test.

    df : dataframe
        All attributes.

    X : array-like of shape (n_samples, n_attributes)
        Selected train attributes.

    y : array-like of shape (n_samples,)
        Target attributes.

    scorer : scoring object
        Scorer to compute on the three methods.

    th : tuple (min, max, step)
        Threshold range.
    """

    def get_scores(model, df_out, X_out, y_out, scorer):
        mid_rc = df.groupby("raceYear")["raceRound"].max().to_numpy() // 2
        get_half = (
            lambda x: f'{x["raceYear"]}{x["raceRound"] <= mid_rc[x["raceYear"] - 2006]}'
        )
        instances_per_half = (
            df.apply(get_half, axis=1).value_counts(sort=False).to_numpy()
        )

        n_splits = len(instances_per_half) - 10
        max_train_size = [instances_per_half[i : 10 + i].sum() for i in range(n_splits)]
        test_size = instances_per_half[10:].tolist()
        tscv = VariableTimeSeriesSplit(
            n_splits=n_splits, max_train_size=max_train_size, test_size=test_size
        )

        s = cross_val_score(model, X_out, y_out, cv=tscv, scoring=scorer, n_jobs=-1)
        return s.sum() / len(s)

    th_min, th_max, th_step = th
    z_scores = np.abs(stats.zscore(X))

    ths = []
    scores = []

    for t in np.arange(th_min, th_max, th_step):
        df_out = df[(z_scores < t).all(axis=1)]
        X_out = X[(z_scores < t).all(axis=1)]
        y_out = y[(z_scores < t).all(axis=1)]
        ths.append(t)
        scores.append(get_scores(model, df_out, X_out, y_out, scorer))

    no_treat = get_scores(model, df, X, y, scorer)

    print("Performance without treatment:", no_treat)
    print("Best performance with treatment:")
    print("     threshold =", max(zip(ths, scores), key=lambda x: x[1])[0])
    print("     performance =", max(scores))

    _, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(
        ths,
        scores,
        color="blue",
        marker="o",
        label="With treatment",
    )
    ax.hlines(
        no_treat,
        xmin=th_min,
        xmax=th_max,
        colors="purple",
        linestyles="--",
        lw=2,
        label="Without treatment",
    )
    ax.set_xticks(np.arange(th_min, th_max))
    ax.title.set_text("Programmatically outlier treatment evolution")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Performance")
    ax.grid()
    ax.legend(loc="lower right")
    plt.show()
