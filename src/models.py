from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    root_mean_squared_error,

    r2_score

)


# Linear Regression Model
def linear_regression_model(X_train, X_test, y_train, y_test):

    """

    Trains a Linear Regression model and makes predictions

    on the test dataset.

    """

    model = LinearRegression()

    # Train the model

    model.fit(X_train, y_train)

    # training vs testing accuracy
    train_r2 = model.score(X_train, y_train)

    test_r2 = model.score(X_test, y_test)

    print(f"Training R² : {train_r2:.4f}")

    print(f"Testing  R² : {test_r2:.4f}")

    # Make predictions

    y_pred = model.predict(X_test)

    print("\n===== LINEAR REGRESSION MODEL =====")

    print("Model trained successfully.")

    print("\nActual Values:")

    print(y_test.head())

    print("\nPredicted Values:")

    print(y_pred[:5])

    print("\n===== FEATURE COEFFICIENTS =====")

    for feature, coefficient in zip(X_train.columns, model.coef_):
        print(f"{feature:<25}: {coefficient:.4f}")
    print(f"\nIntercept: {model.intercept_:.4f}")

    metrics = {

        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "R²": r2_score(y_test, y_pred)

    }
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return model, y_pred, metrics


# Ridge Regression Model
def ridge_regression_model(X_train, X_test, y_train, y_test, alpha=1.0):

    """

    Trains a Ridge Regression model and makes predictions

    on the test dataset.

    """

    model = Ridge(alpha=alpha)

    # Train the model

    model.fit(X_train, y_train)

    # training vs testing accuracy
    train_r2 = model.score(X_train, y_train)

    test_r2 = model.score(X_test, y_test)

    print(f"Training R² : {train_r2:.4f}")

    print(f"Testing  R² : {test_r2:.4f}")

    # Make predictions

    y_pred = model.predict(X_test)

    print("\n===== RIDGE REGRESSION MODEL =====")

    print("Model trained successfully.")

    print("\nActual Values:")

    print(y_test.head())

    print("\nPredicted Values:")

    print(y_pred[:5])

    print("\n===== FEATURE COEFFICIENTS =====")

    for feature, coefficient in zip(X_train.columns, model.coef_):
        print(f"{feature:<25}: {coefficient:.4f}")

    print(f"\nIntercept: {model.intercept_:.4f}")

    metrics = {

        "MAE": mean_absolute_error(y_test, y_pred),

        "MSE": mean_squared_error(y_test, y_pred),

        "RMSE": root_mean_squared_error(y_test, y_pred),

        "R²": r2_score(y_test, y_pred)

    }
    for name, value in metrics.items():

        print(f"{name}: {value:.4f}")

    return model, y_pred, metrics


def baseline_model(y_train, y_test):

    """

    Baseline model that predicts the mean

    sales value for every test sample.

    """

    # Learn the average sales from training data

    baseline_prediction = y_train.mean()

    # Predict the same value for every test sample

    baseline_predictions = [

        baseline_prediction

    ] * len(y_test)

    print("\n===== BASELINE MODEL =====")

    print(f"Average Sales: {baseline_prediction:.2f}")

    metrics = {

        "MAE": mean_absolute_error(y_test, baseline_predictions),

        "MSE": mean_squared_error(y_test, baseline_predictions),

        "RMSE": root_mean_squared_error(y_test, baseline_predictions),

        "R²": r2_score(y_test, baseline_predictions)

    }

    for name, value in metrics.items():

        print(f"{name}: {value:.4f}")

    return baseline_predictions, metrics
