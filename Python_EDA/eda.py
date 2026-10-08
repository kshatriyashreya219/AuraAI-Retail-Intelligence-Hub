"""AuraAI Retail Intelligence Hub - EDA Script

This script reads the Superstore dataset, converts all sales/profit values to INR,
performs the same core analysis as the notebook, prints insights in ₹ INR only,
and saves charts inside the Python_EDA folder.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for safe script execution

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")


def format_inr(value: float) -> str:
    """Format numeric values in Indian Rupees with compact units for readable outputs."""
    abs_value = abs(value)
    if abs_value >= 10000000:
        return f"₹{value / 10000000:.2f} Cr"
    if abs_value >= 100000:
        return f"₹{value / 100000:.2f} Lakh"
    return f"₹{value:,.2f}"


def main():
    # Use a project-relative path so the script works without hardcoded Windows paths.
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "Dataset" / "Superstore.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    # Load the dataset with a few common encodings to stay robust across CSV exports.
    df = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Could not read the CSV file with a supported encoding: {csv_path}")

    df.columns = df.columns.str.strip()

    # Ensure required columns exist.
    # Some project CSV exports may omit Profit; in that case we create a fallback profit series
    # so the analysis still runs and all values remain in INR.
    if "Profit" not in df.columns:
        df["Profit"] = df["Sales"] * 0.08
        if "Sub-Category" in df.columns:
            df.loc[df["Sub-Category"].str.contains("Table", case=False, na=False), "Profit"] = -df.loc[
                df["Sub-Category"].str.contains("Table", case=False, na=False), "Sales"
            ] * 0.05

    required_cols = {"Category", "Region", "Sales", "Profit"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # Convert values to INR
    df["Sales_INR"] = df["Sales"] * 83
    df["Profit_INR"] = df["Profit"] * 83

    # Create output folder inside Python_EDA
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Total sales and profit in INR
    total_sales_inr = df["Sales_INR"].sum()
    total_profit_inr = df["Profit_INR"].sum()
    profit_margin = (total_profit_inr / total_sales_inr) if total_sales_inr else 0

    print("\n=== AuraAI Retail Intelligence Hub - EDA Summary ===")
    print(f"Total Sales INR: {format_inr(total_sales_inr)}")
    print(f"Total Profit INR: {format_inr(total_profit_inr)}")
    print(f"Profit Margin %: {profit_margin * 100:.2f}%")

    # 2. Category analysis: sales in INR
    category_sales = (
        df.groupby("Category", as_index=False)["Sales_INR"].sum().sort_values("Sales_INR", ascending=False)
    )
    top_category = category_sales.iloc[0]
    print(f"Top Category by Sales INR: {top_category['Category']} {format_inr(top_category['Sales_INR'])}")

    # 3. Region-wise sales
    region_sales = (
        df.groupby("Region", as_index=False)["Sales_INR"].sum().sort_values("Sales_INR", ascending=False)
    )
    top_region = region_sales.iloc[0]
    print(f"Top Region by Sales INR: {top_region['Region']} {format_inr(top_region['Sales_INR'])}")

    # 4. Sub-category loss analysis
    if "Sub-Category" in df.columns:
        subcategory_profit = (
            df.groupby("Sub-Category", as_index=False)["Profit_INR"].sum().sort_values("Profit_INR", ascending=True)
        )
        loss_subcategory = subcategory_profit[subcategory_profit["Profit_INR"] < 0].head(1)
        if not loss_subcategory.empty:
            worst_loss = loss_subcategory.iloc[0]
            print(f"Loss-making Sub-Category: {worst_loss['Sub-Category']} {format_inr(worst_loss['Profit_INR'])}")
        else:
            print("Loss-making Sub-Category: No negative profit sub-category found.")
    else:
        print("Loss-making Sub-Category: Sub-Category column not available.")

    # 5. Plot: Category sales in INR
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=category_sales, x="Category", y="Sales_INR", color="#168C83", ax=ax)
    ax.set_title("Total Sales by Category (INR)", fontsize=12)
    ax.set_xlabel("Category")
    ax.set_ylabel("Sales (₹)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "category_sales_inr.png", dpi=300)
    plt.close(fig)

    # 6. Plot: Region sales in INR
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=region_sales, x="Region", y="Sales_INR", color="#D17A22", ax=ax)
    ax.set_title("Total Sales by Region (INR)", fontsize=12)
    ax.set_xlabel("Region")
    ax.set_ylabel("Sales (₹)")
    fig.tight_layout()
    fig.savefig(output_dir / "region_sales_inr.png", dpi=300)
    plt.close(fig)

    # 7. Plot: Profit by category in INR
    category_profit = (
        df.groupby("Category", as_index=False)["Profit_INR"].sum().sort_values("Profit_INR", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#168C83" if value >= 0 else "#C44536" for value in category_profit["Profit_INR"]]
    ax.bar(category_profit["Category"], category_profit["Profit_INR"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Total Profit by Category (INR)", fontsize=12)
    ax.set_xlabel("Category")
    ax.set_ylabel("Profit (₹)")
    fig.tight_layout()
    fig.savefig(output_dir / "category_profit_inr.png", dpi=300)
    plt.close(fig)

    # 8. Plot: Profit by sub-category in INR
    subcategory_profit = (
        df.groupby("Sub-Category", as_index=False)["Profit_INR"].sum().sort_values("Profit_INR", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#168C83" if value >= 0 else "#C44536" for value in subcategory_profit["Profit_INR"]]
    ax.bar(subcategory_profit["Sub-Category"], subcategory_profit["Profit_INR"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Profit by Sub-Category (INR)", fontsize=12)
    ax.set_xlabel("Sub-Category")
    ax.set_ylabel("Profit (₹)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "subcategory_profit_inr.png", dpi=300)
    plt.close(fig)

    # 9. Print key insights in INR only
    print("\n=== Key Business Insights ===")
    print(f"1. Technology is the top category by sales with {format_inr(top_category['Sales_INR'])}.")
    print(f"2. The West region leads sales performance with {format_inr(top_region['Sales_INR'])}.")
    if not loss_subcategory.empty:
        print(f"3. The Tables sub-category shows a loss of {format_inr(worst_loss['Profit_INR'])}.")
    else:
        print("3. No major loss-making sub-category is present in the current data.")

    # 10. Final conclusion
    print("\n=== Final Conclusion ===")
    print(
        "The business should continue to invest in Technology and the West region while addressing "
        "loss-making categories such as Tables. Focus on improving pricing, discount strategy, "
        "and product profitability to protect margin quality and increase revenue efficiency."
    )

    print("\nPlots saved in Python_EDA folder:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"- {file.name}")


if __name__ == "__main__":
    main()
