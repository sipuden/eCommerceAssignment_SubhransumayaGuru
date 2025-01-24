import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")



print(customers.head())
print(products.info())
print(transactions.describe())


merged = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")



merged['TransactionDate'] = pd.to_datetime(merged['TransactionDate'])
sales_trend = merged.groupby('TransactionDate')['TotalValue'].sum()
sales_trend.plot(title="Sales Over Time")
plt.show()




region_sales = merged.groupby('Region')['TotalValue'].sum()
region_sales.plot(kind='bar', title="Sales by Region")
plt.show()
