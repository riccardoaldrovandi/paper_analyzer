import os
import numpy as np
from pysr import PySRRegressor


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard R^2, with an explicit fallback for (near-)constant targets. Plots like
    a flat reference line (e.g. a baseline model with no tunable parameter) produce
    target values with ~zero variance, where the textbook R^2 formula is a 0/0
    division -- reporting that as nan/-inf would wrongly flag a genuinely perfect
    constant fit as broken or overfit.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    scale = max(abs(np.mean(y_true)), 1.0)
    if ss_tot > 1e-8 * (scale ** 2) * len(y_true):
        return 1 - ss_res / ss_tot

    # Near-constant target: fall back to a relative-residual score instead.
    rel_err = np.sqrt(ss_res / len(y_true)) / scale
    return max(0.0, 1.0 - rel_err)


def _select_by_holdout(model: PySRRegressor, X_hold: np.ndarray, y_hold: np.ndarray, tolerance: float = 0.02):
    """
    Scores every equation on PySR's full complexity/loss Pareto front (not just its
    single internally-chosen "best") against points that were held out of the
    search entirely. This measures how well each candidate generalizes instead of
    how well it merely threads through the training points -- which is what
    actually separates a correct simple trend from a flexible equation (e.g. a
    Gaussian-shaped bump from a stray exp(quadratic)) that just happens to fit
    noisy training data well. This is the automated substitute for a human
    eyeballing the fitted curve against the original plot.
    """
    candidates = model.equations_

    scored = []
    for index in candidates.index:
        y_pred = np.asarray(model.predict(X_hold, index=index)).reshape(-1)
        r2 = _r2_score(y_hold, y_pred)
        scored.append((index, r2, candidates.loc[index, "complexity"]))

    best_r2 = max(r2 for _, r2, _ in scored)
    # Among equations within `tolerance` of the best holdout R^2, keep the simplest
    # one (Occam's razor) instead of blindly taking the single top scorer, which
    # could just be a coincidental fluke on a handful of holdout points.
    within_tolerance = [s for s in scored if s[1] >= best_r2 - tolerance]
    best_index, best_r2_final, _ = min(within_tolerance, key=lambda s: s[2])

    return best_index, best_r2_final


def discover_formula(X: np.ndarray, y: np.ndarray, iterations: int = 40) -> dict:
    """
    Applies symbolic regression via PySR to the (X, y) arrays extracted from the plot,
    returning the best mathematical expression discovered together with fit-quality
    metrics. When enough points are available, model selection is driven by a
    held-out validation split rather than PySR's in-sample "best" pick, so an
    equation that merely memorizes noisy, visually-extracted points (instead of
    describing the real trend) gets automatically penalized without requiring a
    human to look at a plot.
    """
    if X is None or y is None or len(X) == 0:
        return {"equation": "No valid data provided for symbolic regression.", "complexity": None, "score": None, "r2": None, "r2_holdout": None}

    # PySR requires that the input X be a 2D matrix (samples x features)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1)

    print(f"[~] Starting PySR (Symbolic Regression) with {iterations} iterations...")

    # Trig/exponential-of-exponential operators are deliberately excluded from the
    # default set: with only ~15-30 noisy, visually-estimated points, high-frequency
    # functions like sin/cos can always thread through the noise and produce a
    # spurious "perfect fit" that models nothing real. "exp"/"log" cover the
    # saturating/asymptotic trends typical of ML performance curves without that risk.
    model = PySRRegressor(
        niterations=iterations,
        binary_operators=["+", "*", "/", "-"],
        unary_operators=["exp", "log"],
        model_selection="best",
        parsimony=0.01,  # penalize complexity more heavily to curb overfitting on sparse data
        # Forbid nesting exp/log inside themselves (e.g. exp(exp(exp(...)))): with so
        # few points that kind of stacking is exactly what lets PySR bend a curve into
        # a bump that threads through the noise instead of the real trend.
        nested_constraints={"exp": {"exp": 0, "log": 0}, "log": {"log": 0, "exp": 0}},
        verbosity=0,  # Reduce the verbose logs of Julia/PySR in the terminal
        # We only ever read the equation/metrics back in-memory via model.equations_,
        # never from disk, so avoid PySR's default of leaving a persistent
        # outputs/<timestamp>/ directory (hall_of_fame.csv + checkpoint.pkl) behind
        # for every single curve fit -- with export-dataset running this per curve
        # across many papers, that clutter adds up fast.
        temp_equation_file=True,
    )

    try:
        n = len(y)
        # With too few points a held-out split would leave nothing meaningful to fit
        # or validate on, so below this size we fall back to PySR's own in-sample
        # "best" pick and flag the result as unvalidated.
        use_holdout = n >= 8

        if use_holdout:
            rng = np.random.default_rng(42)
            idx = rng.permutation(n)
            n_holdout = min(max(3, round(n * 0.25)), n - 5)  # always leave >=5 points to fit on
            holdout_idx, fit_idx = idx[:n_holdout], idx[n_holdout:]
            X_fit, y_fit = X[fit_idx], y[fit_idx]
            X_hold, y_hold = X[holdout_idx], y[holdout_idx]
        else:
            X_fit, y_fit = X, y
            X_hold, y_hold = None, None

        model.fit(X_fit, y_fit)

        if use_holdout:
            best_index, r2_holdout = _select_by_holdout(model, X_hold, y_hold)
        else:
            best_index, r2_holdout = None, None

        best_equation = model.sympy(index=best_index)
        best_row = model.equations_.loc[best_index] if best_index is not None else model.get_best()
        complexity = best_row["complexity"]
        score = best_row["score"]

        # In-sample R^2 (on the points actually used to fit), for reference/transparency.
        y_pred_fit = np.asarray(model.predict(X_fit, index=best_index)).reshape(-1)
        r2_fit = _r2_score(y_fit, y_pred_fit)

        print(f"\n[+] Mathematical formula discovered with success!")
        print(f"    -> Expression: {best_equation}")
        if use_holdout:
            print(f"    -> Complexity: {complexity} | R^2 (fit): {r2_fit:.4f} | R^2 (holdout): {r2_holdout:.4f}")
            if r2_holdout < 0.85 or (r2_fit - r2_holdout) > 0.15:
                print(f"    -> [!] Large fit/holdout gap: this equation may not generalize well, treat with caution.")
        else:
            print(f"    -> Complexity: {complexity} | Score: {score:.4f} | R^2: {r2_fit:.4f}")
            print(f"    -> [!] Only {n} points: too few for a holdout check, result is not cross-validated.")

        return {
            "equation": str(best_equation),
            "complexity": complexity,
            "score": score,
            "r2": r2_fit,
            "r2_holdout": r2_holdout,
        }

    except Exception as e:
        print(f"[-] Error during execution of PySR: {e}")
        return {"equation": "Error in PySR calculation", "complexity": None, "score": None, "r2": None, "r2_holdout": None, "error": str(e)}
