import pandas as pd


# Preprocessing Pipeline
def preprocessing(df):

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(df.info())

    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    print(df.describe(include="all"))

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    # Missing values column-wise
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # Missing values row-wise
    print("\nRows containing missing values:")
    print(df[df.isnull().any(axis=1)])

    print("\nNumber of rows with missing values:",
          df[df.isnull().any(axis=1)].shape[0])

    print("\n" + "=" * 60)
    print("HANDLING MISSING VALUES")
    print("=" * 60)

    # Numerical columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Exclude ID and target columns
    numerical_cols = [
        col for col in numerical_cols
        if col not in ["campaign_id", "sales_lakh"]
    ]

    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].mean())

    print("Missing values after preprocessing:")
    print(df.isnull().sum())

    print("\n" + "=" * 60)

    print("ENCODING CATEGORICAL FEATURES")

    print("=" * 60)

    # region and season are nominal (no inherent order), so one-hot
    # encode them instead of assigning arbitrary ordinal integers.
    # drop_first=True drops one level per column as the baseline
    # category to avoid the dummy variable trap.
    df = pd.get_dummies(
        df,
        columns=["region", "season"],
        prefix=["region", "season"],
        drop_first=True,
        dtype=int
    )

    encoded_cols = [
        col for col in df.columns
        if col.startswith("region_") or col.startswith("season_")
    ]

    print("\nEncoded Dataset Preview:")

    print(df[encoded_cols].head())

    return df
