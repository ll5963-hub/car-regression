"""Used car prices: should the dealer buy this car?

Fits a linear regression of sale price on a 70/30 split of used_car_sales.csv,
compares it with the baseline (training-set average) on the test cars, and
applies both buy rules to the 40 cars in auction_lots.csv.

Run:  pip install pandas statsmodels scikit-learn matplotlib
      python fit.py
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

SEED = 42
PREP_COST = 1500

# ---------------------------------------------------------------- data
df = pd.read_csv("used_car_sales.csv")
print(f"used_car_sales.csv: {df.shape[0]} rows, {df.shape[1]} columns")
train, test = train_test_split(df, test_size=0.3, random_state=SEED)
print(f"Train: {len(train)} cars   Test: {len(test)} cars")

# ---------------------------------------------------------------- fit
full = smf.ols("Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident) + Num_Owners",
               data=train).fit()
print("\n=== Full model (all 5 features) ===")
print(full.summary())

v1 = smf.ols("Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident)",
             data=train).fit()
print("\n=== v1 (Num_Owners dropped) ===")
print(v1.summary())

print("\nAdjusted R-squared: full "
      f"{full.rsquared_adj:.3f}  vs  v1 {v1.rsquared_adj:.3f}")
print("\nv1 coefficients and p-values:")
for term in v1.params.index:
    print(f"  {term:22s} {v1.params[term]:>10,.0f}   p = {v1.pvalues[term]:.4f}")

# ---------------------------------------------------------------- evaluate on test cars
def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def rmse(y, pred):
    return float(np.sqrt(((y - pred) ** 2).mean()))

avg_price = train["Sale_Price"].mean()
y = test["Sale_Price"]
base_pred = pd.Series(avg_price, index=test.index)
v1_pred = v1.predict(test)

print(f"\n=== Test cars ({len(test)}) ===")
print(f"Baseline (average ${avg_price:,.0f}): R2 {r2(y, base_pred):.3f}   "
      f"typical error ${rmse(y, base_pred):,.0f}")
print(f"v1:                          R2 {r2(y, v1_pred):.3f}   "
      f"typical error ${rmse(y, v1_pred):,.0f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y, v1_pred, color="black", s=18, label="v1 regression")
ax.scatter(y, base_pred, color="black", marker="x", s=18, alpha=0.5, label="Baseline (average)")
lims = [min(y.min(), v1_pred.min()) - 1000, max(y.max(), v1_pred.max()) + 1000]
ax.plot(lims, lims, color="gray", lw=1, ls="--", label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual sale price ($)")
ax.set_ylabel("Predicted sale price ($)")
ax.set_title("Test cars: predicted vs actual")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("pred_vs_actual.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------- decision rule
def should_buy(car: dict, auction_price: float) -> str:
    """Buy if auction price + prep cost is below v1's predicted sale price."""
    predicted = float(v1.predict(pd.DataFrame([car]))[0])
    return "buy" if auction_price + PREP_COST < predicted else "pass"


def baseline_should_buy(auction_price: float) -> str:
    return "buy" if auction_price + PREP_COST < avg_price else "pass"


lots = pd.read_csv("auction_lots.csv")
features = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]
lots["profit_if_bought"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - PREP_COST
lots["v1"] = [should_buy(r[features].to_dict(), r["Auction_Price"]) for _, r in lots.iterrows()]
lots["baseline"] = [baseline_should_buy(p) for p in lots["Auction_Price"]]

print(f"\n=== Auction ({len(lots)} lots) ===")
for rule in ["baseline", "v1"]:
    bought = lots[lots[rule] == "buy"]
    losers = (bought["profit_if_bought"] < 0).sum()
    print(f"{rule:9s} buys {len(bought):2d} cars, {losers} lose money, "
          f"total profit ${bought['profit_if_bought'].sum():,.0f}")
lots.to_csv("auction_decisions.csv", index=False)
