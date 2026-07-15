from sklearn.model_selection import cross_val_score


# Cross - Validatian (fold = 5)
def cross_validation(model, X_train, y_train):

    """

    Performs 5-Fold Cross Validation

    and prints the R² score for each fold.

    """

    scores = cross_val_score(

        estimator=model,

        X=X_train,

        y=y_train,

        cv=5,

        scoring="r2"

    )

    print("\n===== 5-FOLD CROSS VALIDATION =====")

    for i, score in enumerate(scores, start=1):

        print(f"Fold {i}: {score:.4f}")

    print(f"\nAverage R² : {scores.mean():.4f}")

    print(f"Std Dev    : {scores.std():.4f}")

    return scores


# Compare Linear vs Ridge
def compare_models(linear_metrics, ridge_metrics):

    """

    Prints a side-by-side comparison of Linear Regression

    and Ridge Regression metrics.

    """

    print("\n===== LINEAR vs RIDGE COMPARISON =====")

    print(f"{'Metric':<10}{'Linear':>12}{'Ridge':>12}")

    for name in linear_metrics:

        print(f"{name:<10}{linear_metrics[name]:>12.4f}{ridge_metrics[name]:>12.4f}")

    return None
