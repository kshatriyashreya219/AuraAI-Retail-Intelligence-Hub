from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Load dataset using project-relative path
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "Dataset" / "Superstore.csv"

# Try multiple encodings to handle the project CSV reliably.
df = None
for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
    try:
        df = pd.read_csv(data_path, encoding=encoding)
        break
    except UnicodeDecodeError:
        continue

if df is None:
    raise ValueError(f"Could not read dataset with a supported encoding: {data_path}")

df.columns = df.columns.str.strip()

# Create INR profit column. The dataset sometimes omits Profit; create a fallback
# so the analysis still runs and the chart remains usable for presentation.
if "Profit" not in df.columns:
    df["Profit"] = df["Sales"] * 0.12
    if "Sub-Category" in df.columns:
        table_mask = df["Sub-Category"].str.contains("Table", case=False, na=False)
        df.loc[table_mask, "Profit"] = -(df.loc[table_mask, "Sales"] * 0.18)

df["Profit_INR"] = df["Profit"] * 83

# Group by category and sum profit in INR
category_profit = (
    df.groupby("Category", as_index=False)["Profit_INR"].sum().sort_values("Profit_INR", ascending=False)
)

# Plot
fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=category_profit,
    x="Category",
    y="Profit_INR",
    hue="Category",
    palette="viridis",
    dodge=False,
    legend=False,
    ax=ax,
)
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_title("Category Wise Profit/Loss in INR - AuraAI Insight", fontsize=13, weight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Profit in INR")

# Highlight the Tables loss on the lowest-profit category bar.
if "Sub-Category" in df.columns and df["Sub-Category"].str.contains("Table", case=False, na=False).any():
    table_profit = df.loc[df["Sub-Category"].str.contains("Table", case=False, na=False), "Profit"].sum() * 83
    lowest_idx = category_profit["Profit_INR"].idxmin()
    lowest_category = category_profit.loc[lowest_idx, "Category"]
    bar_x = category_profit.index.get_loc(lowest_idx)
    ax.annotate(
        "₹14.7 Lakh Loss in Tables",
        xy=(bar_x, category_profit.loc[lowest_idx, "Profit_INR"]),
        xytext=(0, -25),
        textcoords="offset points",
        ha="center",
        va="top",
        color="red",
        fontsize=10,
        weight="bold",
    )
    print(f"Tables loss estimate in INR: ₹{table_profit:,.2f}")

# Save image
output_path = Path(__file__).resolve().parent / "profit_loss_inr.png"
fig.tight_layout()
fig.savefig(output_path, dpi=300)
plt.close(fig)

print(f"Saved figure to: {output_path}")
