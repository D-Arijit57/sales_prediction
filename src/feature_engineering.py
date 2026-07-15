# Feature Engineering
def feature_engineering(df):

    """

    Creates new features derived from the existing

    marketing, region and salesperson columns.

    """

    print("\n" + "=" * 60)

    print("FEATURE ENGINEERING")

    print("=" * 60)

    # Total spend across all marketing channels
    df["total_marketing_budget"] = (
        df["tv_budget"] + df["radio_budget"] + df["newspaper_budget"]
    )

    # Share of the total budget spent on TV
    df["tv_budget_percentage"] = (
        df["tv_budget"] / df["total_marketing_budget"] * 100
    )

    # region is now one-hot encoded (one column per non-baseline region),
    # so the interaction with each region dummy is added separately.
    # This lets the model learn a distinct TV/newspaper effect per region
    # instead of forcing a single linear relationship across an
    # arbitrarily ordered region scale.
    region_dummy_cols = [
        col for col in df.columns if col.startswith("region_")
    ]

    interaction_cols = []

    for region_col in region_dummy_cols:

        newspaper_col = f"newspaper_{region_col}_interaction"
        tv_col = f"tv_{region_col}_interaction"

        df[newspaper_col] = df["newspaper_budget"] * df[region_col]
        df[tv_col] = df["tv_budget"] * df[region_col]

        interaction_cols += [newspaper_col, tv_col]

    # Flag salespeople at or above the median experience level
    experience_threshold = df["salesperson_experience"].median()

    df["experienced_salesperson"] = (
        df["salesperson_experience"] >= experience_threshold
    ).astype(int)

    print(f"\nExperience threshold (median): {experience_threshold}")

    print("\nEngineered Features Preview:")

    print(df[[
        "total_marketing_budget",
        "tv_budget_percentage",
        *interaction_cols,
        "experienced_salesperson"
    ]].head())

    return df
