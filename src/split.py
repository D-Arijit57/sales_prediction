import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Train - Test for the data
def train_test_data(df):

    """

    Splits the dataset into training and testing sets.

    """

    # Features: everything except the ID and target columns.
    # region/season are now expanded into one-hot dummy columns, and
    # the interaction columns are named per region dummy, so the
    # feature set is built dynamically instead of a fixed list.
    exclude_cols = ["campaign_id", "sales_lakh"]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]

    # Target

    y = df["sales_lakh"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )

    print("\n===== TRAIN TEST SPLIT =====")

    print(f"X_train : {X_train.shape}")

    print(f"X_test  : {X_test.shape}")

    print(f"y_train : {y_train.shape}")

    print(f"y_test  : {y_test.shape}")

    # Scale features: fit on the training set only, then apply the
    # same transform to the test set to avoid leaking test statistics.
    scaler = MinMaxScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return (
        X_train,

        X_test,

        y_train,

        y_test

    )
