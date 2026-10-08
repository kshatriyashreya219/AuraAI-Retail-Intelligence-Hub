"""Calculate Aura retail-performance metrics and product recommendations."""

def calculate_aura_metrics(data):
    low_aura = data.nsmallest(5, "Aura")[["Product Name", "Aura", "Sales"]]

    return {
        "total_sales": data["Sales"].sum(),
        "total_profit": data["Profit"].sum() if "Profit" in data.columns else None,
        "avg_aura": data["Aura"].mean(),
        "low_aura": low_aura,
    }


def get_recommendations(aura_score):
    if aura_score < 10:
        return "Low Aura: Review pricing and costs to improve profitability."
    if aura_score < 30:
        return "Medium Aura: Look for opportunities to improve profit margins."
    return "High Aura: Maintain the current performance and monitor trends."