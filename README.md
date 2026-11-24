# Used car prices: should the dealer buy this car?

Linear regression of lot sale price (statsmodels OLS) on a 70/30 split of `used_car_sales.csv`. The 40 cars in `auction_lots.csv` are used to test the buy rule.

```
pip install pandas statsmodels scikit-learn matplotlib
python fit.py
```

`fit.py` prints every number on the slides (saved in `output.txt`). It also writes `pred_vs_actual.png`, the chart on the slides, and `auction_decisions.csv`, the buy/pass call for each auction lot.
