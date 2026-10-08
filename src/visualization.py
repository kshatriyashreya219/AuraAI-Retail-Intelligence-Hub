import plotly.express as px


def plot_aura_distribution(data):
	required_columns = {"Aura"}
	if not required_columns.issubset(data.columns):
		return None

	return px.histogram(
		data,
		x="Aura",
		nbins=50,
		title="Aura Distribution",
		labels={"Aura": "Aura (%)"},
	)


def plot_sales_vs_profit(data):
	required_columns = {"Profit", "Sales", "Aura", "Product Name"}
	if not required_columns.issubset(data.columns):
		return None

	return px.scatter(
		data,
		x="Sales",
		y="Profit",
		color="Aura",
		hover_data=["Product Name"],
		title="Sales vs Profit",
	)


def plot_top_products(data):
	required_columns = {"Sales", "Product Name"}
	if not required_columns.issubset(data.columns):
		return None

	top_products = (
		data.groupby("Product Name", as_index=False)["Sales"]
		.sum()
		.nlargest(10, "Sales")
	)
	figure = px.bar(
		top_products,
		x="Sales",
		y="Product Name",
		orientation="h",
		title="Top 10 Products by Sales",
	)
	figure.update_yaxes(categoryorder="total ascending")
	return figure
