# CloudCost: AWS EC2 Cost Predictor

A Streamlit app that estimates the hourly and total cost of an AWS EC2 instance from its configuration, using a Random Forest trained on AWS's public EC2 price list.

It predicts **list prices** for EC2 compute. It does not forecast future spending, and it does not include S3, RDS, EBS volume charges or data transfer.

## How it works

```
training.py   AmazonEC2.csv → filter → sample 50,000 rows → feature engineering
              → train pipeline → evaluate → save .pkl files

app.py        form inputs → pipeline predicts hourly price → hourly price × hours
```

- **Data:** 7.66M rows, filtered to 3.17M hourly compute rows with a price above zero, then randomly sampled to 50,000 (80/20 train/test split).
- **Features:** 24 inputs (9 numeric, 15 categorical), which become 1,349 columns after one-hot encoding.
- **Model:** scikit-learn pipeline with median and constant imputation, one-hot encoding and a 200-tree `RandomForestRegressor`.

## Results

Measured on the 10,000-row test set, with prices in USD per hour.

| Metric | Test |
|---|---|
| R² | 0.990 (train: 0.997) |
| MAE | $0.46 |
| RMSE | $2.19 |
| Median % error | 5.2% |
| Mean % error | 9.9% |

Always predicting the average price gives an MAE of $8.62.

## Run it

Tested with Python 3.11. Requires [Git LFS](https://git-lfs.com) because the model file is 720 MB. Loading it uses about 1.2 GB of RAM.

```bash
git lfs install
git clone https://github.com/Snekhagopal/AWS_Cost_predictor.git
cd AWS_Cost_predictor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Retrain (optional)

1. Download the [EC2 price list CSV](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/index.csv).
2. Save it as `data/AmazonEC2.csv`, with the column header as the first line. Remove any metadata lines above it.
3. Run `python training.py`.

## Limitations

- **EBS throughput is not auto-filled.** Other specs fill in from the selected instance type, but this one stays at 0 unless you enter it.
- **Reserved pricing is incomplete.** All Upfront is not covered, and Partial Upfront leaves out the upfront fee.
- **Some sidebar items are placeholders.** The greyed-out items are not built yet.
- **Metrics on the page are fixed text.** They were copied from the evaluation above and are not calculated live.
