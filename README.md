# Sales Prediction

A small regression pipeline that predicts campaign sales (in lakhs) from
marketing spend, discounting, salesperson experience, region, and season.

## Dataset

`sales_prediction_dataset.json` — 300 marketing campaign records with the
following fields:

| Column                   | Description                                  |
|--------------------------|-----------------------------------------------|
| `campaign_id`            | Unique campaign identifier                    |
| `tv_budget`              | TV advertising spend                          |
| `radio_budget`           | Radio advertising spend                       |
| `newspaper_budget`       | Newspaper advertising spend                   |
| `discount_percent`       | Discount offered during the campaign          |
| `salesperson_experience` | Years of experience of the assigned salesperson |
| `region`                 | Campaign region (North/South/East/West)       |
| `season`                 | Season the campaign ran in                    |
| `sales_lakh`             | Target: sales generated, in lakhs             |

## Project structure

```
Sales_prediction/
├── main.py                     # entry point — runs the full pipeline
├── sales_prediction_dataset.json
└── src/
    ├── data_loader.py          # loads the dataset into a DataFrame
    ├── preprocessing.py        # missing-value imputation, one-hot encoding
    ├── feature_engineering.py  # derived budget/interaction/experience features
    ├── split.py                # train/test split + MinMax feature scaling
    ├── models.py                # Linear Regression, Ridge Regression, baseline
    └── evaluation.py           # 5-fold cross-validation, model comparison
```

## Pipeline

1. **Preprocessing** — imputes missing numeric values with the column mean
   and one-hot encodes `region` and `season`.
2. **Feature engineering** — adds total marketing budget, TV budget share,
   per-region TV/newspaper spend interactions, and an "experienced
   salesperson" flag (at/above median experience).
3. **Split & scale** — an 80/20 train/test split, then `MinMaxScaler` fit on
   the training set and applied to both sets.
4. **Modeling** — trains Linear Regression and Ridge Regression, and a mean
   baseline for comparison.
5. **Evaluation** — 5-fold cross-validation on the linear model, plus a
   side-by-side MAE/MSE/RMSE/R² comparison of Linear vs. Ridge.

## Requirements

- Python 3
- pandas
- scikit-learn

```bash
pip install pandas scikit-learn
```

## Usage

```bash
python main.py
```
