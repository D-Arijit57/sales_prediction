from src.data_loader import load_data
from src.preprocessing import preprocessing
from src.feature_engineering import feature_engineering
from src.split import train_test_data
from src.models import linear_regression_model, ridge_regression_model, baseline_model
from src.evaluation import cross_validation, compare_models


def main():
    df = load_data()

    df = preprocessing(df)

    df = feature_engineering(df)

    X_train, X_test, y_train, y_test = train_test_data(df)

    model, y_pred, linear_metrics = linear_regression_model(
        X_train,
        X_test,
        y_train,
        y_test)

    ridge_model, ridge_pred, ridge_metrics = ridge_regression_model(
        X_train,
        X_test,
        y_train,
        y_test)

    baseline_model(
        y_train,
        y_test)

    cross_validation(
        model,
        X_train,
        y_train
    )

    compare_models(linear_metrics, ridge_metrics)


if __name__ == "__main__":
    main()
