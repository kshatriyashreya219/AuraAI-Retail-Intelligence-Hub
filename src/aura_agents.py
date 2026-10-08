"""Coordinate sequential CrewAI analysis for AURA OS retail operations.

Pandas-backed tools analyze profit and loss, estimate product demand, and detect
duplicate product assignments within Shaadi Code wedding orders. CrewAI assigns
the results to financial, inventory, and wedding-planning agents.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool


def _load_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError("The uploaded orders CSV could not be found.")

    try:
        dataframe = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        dataframe = pd.read_csv(path, encoding="cp1252")
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    if dataframe.empty:
        raise ValueError("The uploaded orders CSV contains no order rows.")
    return dataframe


def _column(dataframe: pd.DataFrame, *candidates: str) -> str | None:
    columns_by_name = {column.casefold(): column for column in dataframe.columns}
    return next((columns_by_name[name.casefold()] for name in candidates if name.casefold() in columns_by_name), None)


def _clean_records(dataframe: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    return dataframe.head(limit).where(pd.notna(dataframe), None).to_dict(orient="records")


@tool("Pandas Profit Analysis")
def profit_analysis_tool(csv_path: str) -> str:
    """Analyze order-level profit and loss from a CSV path using pandas."""
    try:
        dataframe = _load_csv(csv_path)
        profit_column = _column(dataframe, "Profit", "Profit_INR")
        product_column = _column(dataframe, "Product Name", "Product", "Item")
        date_column = _column(dataframe, "Order Date", "Date", "Sale Date")
        sales_column = _column(dataframe, "Sales", "Sales_INR", "Revenue")

        if profit_column is None:
            return json.dumps({
                "status": "insufficient_data",
                "message": "The CSV has no Profit or Profit_INR column; loss causes cannot be determined from sales alone.",
                "available_columns": list(dataframe.columns),
            })

        dataframe[profit_column] = pd.to_numeric(dataframe[profit_column], errors="coerce")
        loss_rows = dataframe.loc[dataframe[profit_column] < 0].copy()
        if loss_rows.empty:
            return json.dumps({
                "status": "success",
                "message": "No loss-making rows were found in the uploaded data.",
                "loss_row_count": 0,
                "total_profit": round(float(dataframe[profit_column].sum()), 2),
            })

        group_columns = [column for column in (product_column,) if column is not None]
        if group_columns:
            product_losses = (
                loss_rows.groupby(group_columns, dropna=False)[profit_column]
                .agg(loss="sum", loss_rows="count")
                .sort_values("loss")
                .head(10)
                .reset_index()
                .to_dict(orient="records")
            )
        else:
            product_losses = []

        evidence_columns = [column for column in (date_column, product_column, sales_column, profit_column) if column]
        return json.dumps({
            "status": "success",
            "loss_row_count": int(len(loss_rows)),
            "total_loss": round(float(loss_rows[profit_column].sum()), 2),
            "total_profit": round(float(dataframe[profit_column].sum()), 2),
            "largest_product_losses": product_losses,
            "loss_examples": _clean_records(loss_rows[evidence_columns].sort_values(profit_column).head(10)),
            "diagnostic_limit": "The CSV shows loss patterns but does not include cost drivers, so it cannot establish a causal explanation beyond the available fields.",
        }, default=str)
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)})


@tool("Demand Forecast")
def demand_forecast_tool(csv_path: str) -> str:
    """Estimate 30-day product demand from dated order rows and available stock."""
    try:
        dataframe = _load_csv(csv_path)
        product_column = _column(dataframe, "Product Name", "Product", "Item", "SKU")
        date_column = _column(dataframe, "Order Date", "Date", "Sale Date")
        quantity_column = _column(dataframe, "Quantity", "Units", "Units Sold")
        stock_column = _column(dataframe, "Available Stock", "On Hand", "Stock", "Inventory")

        if product_column is None:
            return json.dumps({"status": "insufficient_data", "message": "A product or SKU column is required to forecast demand."})
        if date_column is None:
            return json.dumps({"status": "insufficient_data", "message": "A date column is required to calculate the forecast period."})

        raw_dates = dataframe[date_column].astype("string").str.strip()
        date_parts = raw_dates.str.extract(r"^(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})")
        first_component = pd.to_numeric(date_parts[0], errors="coerce")
        second_component = pd.to_numeric(date_parts[1], errors="coerce")
        year_first = first_component.ge(1000).fillna(False).mean() >= 0.5
        day_first = not year_first and bool(
            ((first_component > 12) & (first_component <= 31) & (second_component <= 12)).any()
        )
        dataframe[date_column] = pd.to_datetime(raw_dates, errors="coerce", dayfirst=day_first)
        dataframe = dataframe.dropna(subset=[date_column])
        if dataframe.empty:
            return json.dumps({"status": "insufficient_data", "message": "No valid dates were found in the CSV."})

        if quantity_column:
            dataframe[quantity_column] = pd.to_numeric(dataframe[quantity_column], errors="coerce").fillna(0)
            demand_column = quantity_column
            demand_unit = "units"
        else:
            demand_column = "_demand_rows"
            dataframe[demand_column] = 1
            demand_unit = "order lines (quantity was not provided)"

        observed_days = max((dataframe[date_column].max() - dataframe[date_column].min()).days + 1, 1)
        grouped = dataframe.groupby(product_column, dropna=False)[demand_column].sum().rename("observed_demand").to_frame()
        grouped["forecast_30_days"] = grouped["observed_demand"] / observed_days * 30
        grouped["demand_unit"] = demand_unit

        if stock_column:
            dataframe[stock_column] = pd.to_numeric(dataframe[stock_column], errors="coerce")
            stock = dataframe.groupby(product_column, dropna=False)[stock_column].last()
            grouped["available_stock"] = stock
            grouped["suggested_reorder"] = (grouped["forecast_30_days"] - grouped["available_stock"].fillna(0)).clip(lower=0)

        forecasts = []
        for product, row in grouped.sort_values("forecast_30_days", ascending=False).head(15).iterrows():
            forecast = {
                "product": str(product),
                "observed_demand": round(float(row["observed_demand"]), 2),
                "forecast_30_days": round(float(row["forecast_30_days"]), 2),
                "demand_unit": demand_unit,
            }
            if stock_column:
                forecast["available_stock"] = None if pd.isna(row["available_stock"]) else float(row["available_stock"])
                forecast["suggested_reorder"] = round(float(row["suggested_reorder"]), 2)
            forecasts.append(forecast)

        return json.dumps({
            "status": "success",
            "observed_period_days": observed_days,
            "forecast_horizon_days": 30,
            "stock_data_available": stock_column is not None,
            "forecast_method": "Average historical daily demand extrapolated over 30 days; this is a baseline forecast, not a seasonality-adjusted model.",
            "products": forecasts,
        })
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)})


@tool("Shaadi Code Anti-Clash Detector")
def shaadi_clash_tool(input_data: str) -> str:
    """Detect duplicate products assigned to the same Shaadi Code in a CSV."""
    try:
        parameters = json.loads(input_data)
        dataframe = _load_csv(parameters["csv_path"])
        code_column = _column(dataframe, "Shaadi Code", "Wedding Code", "Event Code")
        product_column = _column(dataframe, "Product Name", "Product", "Item", "SKU")
        requested_code = parameters.get("shaadi_code")

        if code_column is None or product_column is None:
            return json.dumps({
                "status": "insufficient_data",
                "message": "The CSV must contain a Shaadi Code (or Wedding Code/Event Code) and a product column to detect clashes.",
                "available_columns": list(dataframe.columns),
            })

        scoped = dataframe
        if requested_code:
            scoped = dataframe.loc[dataframe[code_column].astype(str).str.casefold() == str(requested_code).casefold()]
        clashes = (
            scoped.groupby([code_column, product_column], dropna=False)
            .size()
            .rename("order_count")
            .reset_index()
        )
        clashes = clashes.loc[clashes["order_count"] > 1].sort_values("order_count", ascending=False)
        return json.dumps({
            "status": "success",
            "shaadi_code": requested_code,
            "codes_checked": int(scoped[code_column].nunique()),
            "clash_count": int(len(clashes)),
            "anti_clashes": clashes.to_dict(orient="records"),
            "message": "No duplicate product assignments were detected." if clashes.empty else "Repeated product assignments were detected for the same Shaadi Code.",
        }, default=str)
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)})


def _build_crew(csv_path: str, shaadi_code: str | None) -> Crew:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY must be set before running the CrewAI agents.")

    llm = LLM(model="openai/gpt-4o", temperature=0.2)
    profit_agent = Agent(
        role="Financial Analyst",
        goal="Identify evidence-based loss patterns and explain only what the uploaded CSV supports.",
        backstory="You are a careful retail financial analyst. You distinguish observed correlations from proven causes.",
        tools=[profit_analysis_tool],
        llm=llm,
        verbose=False,
    )
    stock_agent = Agent(
        role="Inventory Manager",
        goal="Estimate product demand and recommend evidence-based reorder priorities.",
        backstory="You plan inventory from historical demand and clearly label assumptions and missing stock data.",
        tools=[demand_forecast_tool],
        llm=llm,
        verbose=False,
    )
    wedding_agent = Agent(
        role="Wedding Planner",
        goal="Review Shaadi Code bulk orders and detect same-product assignment conflicts.",
        backstory="You coordinate wedding bulk orders and flag repeated product allocations within the same event code.",
        tools=[shaadi_clash_tool],
        llm=llm,
        verbose=False,
    )

    profit_task = Task(
        description=(
            f"Analyze the CSV at {csv_path} using the Pandas Profit Analysis tool. "
            "Summarize why losses occurred only to the extent supported by the available columns. "
            "Call out missing profit data and distinguish loss patterns from causal factors."
        ),
        expected_output="A concise financial analysis with loss totals, affected products, evidence, and data limitations.",
        agent=profit_agent,
    )
    stock_task = Task(
        description=(
            f"Analyze the CSV at {csv_path} using the Demand Forecast tool. "
            "Recommend products to prioritize for reorder, give the 30-day demand estimate, "
            "and state whether quantity and stock fields were available."
        ),
        expected_output="A concise demand forecast and reorder recommendation with assumptions and units.",
        agent=stock_agent,
        context=[profit_task],
    )
    wedding_input = json.dumps({"csv_path": csv_path, "shaadi_code": shaadi_code})
    wedding_task = Task(
        description=(
            "Use the Shaadi Code Anti-Clash Detector tool with this input: "
            f"{wedding_input}. Report duplicate products per code. "
            "If the required fields are absent, state that clash detection was not possible."
        ),
        expected_output="A concise wedding bulk-order report with detected product clashes or explicit missing-data status.",
        agent=wedding_agent,
        context=[profit_task, stock_task],
    )

    return Crew(
        agents=[profit_agent, stock_agent, wedding_agent],
        tasks=[profit_task, stock_task, wedding_task],
        process=Process.sequential,
        verbose=False,
    )


def run_aura_agents(csv_path: str, shaadi_code: str | None = None) -> dict[str, str]:
    """Run all three AURA OS agents in sequential order and return their reports."""
    crew = _build_crew(csv_path, shaadi_code)
    result = crew.kickoff()
    task_outputs = getattr(result, "tasks_output", [])
    if len(task_outputs) < 3:
        raise RuntimeError("CrewAI did not return output for all three agents.")
    return {
        "profit_agent": str(task_outputs[0].raw),
        "stock_agent": str(task_outputs[1].raw),
        "wedding_agent": str(task_outputs[2].raw),
    }
