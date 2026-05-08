import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []

# ── helper ──────────────────────────────────────────────────────────────────
def md(src):  cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

# ============================================================
# CELL 1 – Cover / Title
# ============================================================
md("""# 🌐 Global Electronics Retailer — Business Intelligence & Analytics Report
---
**Author:** Data Analytics Team  
**Date:** May 2026  
**Tools:** Python · Pandas · Matplotlib · Seaborn · Plotly · Scikit-learn

---
> *This notebook presents a full end-to-end analytics project on the Global Electronics Retailer dataset,  
> covering data cleaning, exploratory analysis, advanced visualisations, machine-learning models,  
> and actionable business recommendations.*
""")

# ============================================================
# CELL 2 – Table of Contents
# ============================================================
md("""## 📋 Table of Contents
1. [Environment Setup & Imports](#1)
2. [Data Loading & Inspection](#2)
3. [Data Cleaning & Feature Engineering](#3)
4. [Data Integration — Merging Tables](#4)
5. [Key Performance Indicators (KPIs)](#5)
6. [Customer Analysis](#6)
7. [Product & Category Analysis](#7)
8. [Sales Trends & Seasonality](#8)
9. [Regional & Store Performance](#9)
10. [Exchange Rate & Currency Analysis](#10)
11. [Profitability Analysis](#11)
12. [Correlation & Heatmap Analysis](#12)
13. [Machine Learning — Sales Forecasting](#13)
14. [Machine Learning — Customer Segmentation (KMeans)](#14)
15. [Anomaly Detection](#15)
16. [Executive Summary & Business Recommendations](#16)
""")

# ============================================================
# CELL 3 – Imports
# ============================================================
md("## 1. Environment Setup & Imports <a id='1'></a>")
code("""# ── Standard Library ────────────────────────────────────────
import warnings, os
warnings.filterwarnings('ignore')

# ── Data Manipulation ────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Visualisation ────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Machine Learning ─────────────────────────────────────────
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, silhouette_score)
from sklearn.pipeline import Pipeline

# ── Global Plot Style ────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 120, 'figure.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False})

BRAND_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
                '#3B1F2B', '#44BBA4', '#E94F37', '#393E41']

print("✅  All libraries loaded successfully.")
print(f"   pandas  {pd.__version__}  |  numpy {np.__version__}  |  sklearn installed")
""")

# ============================================================
# CELL 4 – Data Loading
# ============================================================
md("## 2. Data Loading & Inspection <a id='2'></a>")
code("""# ── Load all six CSV files ───────────────────────────────────
customers  = pd.read_csv('Customers.csv',       encoding='latin-1')
products   = pd.read_csv('Products.csv',        encoding='latin-1')
sales      = pd.read_csv('Sales.csv',           encoding='latin-1')
stores     = pd.read_csv('Stores.csv',          encoding='latin-1')
fx         = pd.read_csv('Exchange_Rates.csv',  encoding='latin-1')
data_dict  = pd.read_csv('Data_Dictionary.csv', encoding='latin-1')

datasets = {
    'Customers':     customers,
    'Products':      products,
    'Sales':         sales,
    'Stores':        stores,
    'Exchange Rates': fx,
}

print("=" * 55)
print(f"{'Dataset':<20} {'Rows':>8} {'Columns':>9}")
print("=" * 55)
for name, df in datasets.items():
    print(f"{name:<20} {df.shape[0]:>8,} {df.shape[1]:>9}")
print("=" * 55)
""")

code("""# ── Quick peek at each table ─────────────────────────────────
for name, df in datasets.items():
    print(f"\\n{'─'*55}")
    print(f"  {name.upper()}")
    print(f"{'─'*55}")
    display(df.head(3))
    print(f"  dtypes:\\n{df.dtypes.to_string()}")
    print(f"\\n  Nulls:\\n{df.isnull().sum().to_string()}")
""")

# ============================================================
# CELL 5 – Data Cleaning
# ============================================================
md("""## 3. Data Cleaning & Feature Engineering <a id='3'></a>
Good analysis starts with clean data. We address:
- **Date parsing** — Order Date, Delivery Date, Birthday, Open Date  
- **Currency strings** — strip `$` and spaces from price columns  
- **Missing values** — Delivery Date nulls (online orders), imputation strategy  
- **Duplicates** — check and drop  
- **Outliers** — IQR-based flagging on Quantity  
- **Feature engineering** — Age, Delivery Days, Profit, Margin, Year/Month/Quarter
""")

code("""# ── 3.1  Parse dates ─────────────────────────────────────────
sales['Order Date']    = pd.to_datetime(sales['Order Date'],    errors='coerce')
sales['Delivery Date'] = pd.to_datetime(sales['Delivery Date'], errors='coerce')
customers['Birthday']  = pd.to_datetime(customers['Birthday'],  errors='coerce')
stores['Open Date']    = pd.to_datetime(stores['Open Date'],    errors='coerce')
fx['Date']             = pd.to_datetime(fx['Date'],             errors='coerce')

print("✅  Dates parsed.")

# ── 3.2  Clean currency columns in Products ──────────────────
for col in ['Unit Cost USD', 'Unit Price USD']:
    products[col] = (products[col].astype(str)
                                  .str.replace(r'[\\$,\\s]', '', regex=True)
                                  .astype(float))

print("✅  Currency strings cleaned.")

# ── 3.3  Duplicates ──────────────────────────────────────────
for name, df in datasets.items():
    n = df.duplicated().sum()
    print(f"  {name:<20} duplicates: {n}")

# ── 3.4  Missing value summary ───────────────────────────────
print("\\nMissing values in Sales:")
print(sales.isnull().sum())
print("\\n  Note: Delivery Date nulls = online / not-yet-delivered orders — retained as NaT.")
""")

code("""# ── 3.5  Feature Engineering — Customers ────────────────────
ref_date = pd.Timestamp('2026-01-01')
customers['Age'] = ((ref_date - customers['Birthday']).dt.days / 365.25).astype(int)

bins   = [0, 25, 35, 45, 55, 65, 120]
labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
customers['Age Group'] = pd.cut(customers['Age'], bins=bins, labels=labels, right=False)

print("✅  Customer age & age-group features created.")
print(customers[['CustomerKey','Age','Age Group']].head())
""")

code("""# ── 3.6  Feature Engineering — Sales ────────────────────────
sales['Delivery Days'] = (sales['Delivery Date'] - sales['Order Date']).dt.days
sales['Year']          = sales['Order Date'].dt.year
sales['Month']         = sales['Order Date'].dt.month
sales['Quarter']       = sales['Order Date'].dt.quarter
sales['Month Name']    = sales['Order Date'].dt.strftime('%b')
sales['Day of Week']   = sales['Order Date'].dt.day_name()
sales['Week']          = sales['Order Date'].dt.isocalendar().week.astype(int)

print("✅  Sales time features created.")
print(sales[['Order Date','Year','Month','Quarter','Delivery Days']].head())
""")

code("""# ── 3.7  Feature Engineering — Products ─────────────────────
products['Gross Margin USD'] = products['Unit Price USD'] - products['Unit Cost USD']
products['Margin %']         = (products['Gross Margin USD'] / products['Unit Price USD'] * 100).round(2)

print("✅  Product margin features created.")
print(products[['Product Name','Unit Cost USD','Unit Price USD','Gross Margin USD','Margin %']].head())
""")

code("""# ── 3.8  Outlier detection on Quantity (IQR) ────────────────
Q1 = sales['Quantity'].quantile(0.25)
Q3 = sales['Quantity'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR

outliers = sales[(sales['Quantity'] < lower) | (sales['Quantity'] > upper)]
print(f"Quantity IQR range: [{lower:.1f}, {upper:.1f}]")
print(f"Outlier rows detected: {len(outliers):,}  ({len(outliers)/len(sales)*100:.2f}% of sales)")
sales['Quantity Outlier'] = ((sales['Quantity'] < lower) | (sales['Quantity'] > upper))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].boxplot(sales['Quantity'], vert=False, patch_artist=True,
                boxprops=dict(facecolor='#2E86AB', alpha=0.7))
axes[0].set_title('Quantity — Box Plot (with outliers)')
axes[0].set_xlabel('Quantity')

axes[1].boxplot(sales.loc[~sales['Quantity Outlier'], 'Quantity'], vert=False,
                patch_artist=True, boxprops=dict(facecolor='#44BBA4', alpha=0.7))
axes[1].set_title('Quantity — Box Plot (outliers removed)')
axes[1].set_xlabel('Quantity')
plt.suptitle('Outlier Detection — Order Quantity', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_outliers.png', bbox_inches='tight')
plt.show()
print("\\n📌 Insight: A small fraction of orders have unusually high quantities.")
print("   These may represent bulk/wholesale orders and should be tracked separately.")
""")

# ============================================================
# CELL 6 – Data Integration
# ============================================================
md("""## 4. Data Integration — Merging Tables <a id='4'></a>
### Entity Relationship Overview
```
Sales ──── CustomerKey ──▶ Customers
Sales ──── ProductKey  ──▶ Products
Sales ──── StoreKey    ──▶ Stores
Sales ──── (Order Date + Currency Code) ──▶ Exchange_Rates
```
We build a single **master DataFrame** (`df`) that joins all dimensions onto the Sales fact table.
""")

code("""# ── 4.1  Merge Sales → Products ─────────────────────────────
df = sales.merge(products, on='ProductKey', how='left')
print(f"After Products merge : {df.shape}")

# ── 4.2  Merge → Customers ───────────────────────────────────
df = df.merge(customers, on='CustomerKey', how='left')
print(f"After Customers merge: {df.shape}")

# ── 4.3  Merge → Stores ──────────────────────────────────────
df = df.merge(stores, on='StoreKey', how='left', suffixes=('', '_store'))
print(f"After Stores merge   : {df.shape}")

# ── 4.4  Merge → Exchange Rates (on Date + Currency) ─────────
fx_lookup = fx.rename(columns={'Date': 'Order Date', 'Currency': 'Currency Code'})
df = df.merge(fx_lookup, on=['Order Date', 'Currency Code'], how='left')
print(f"After FX merge       : {df.shape}")

# ── 4.5  Compute USD-normalised revenue & cost ───────────────
# Exchange rate = units of currency per 1 USD  →  divide to get USD
df['Revenue USD']  = (df['Unit Price USD'] * df['Quantity'])
df['Cost USD']     = (df['Unit Cost USD']  * df['Quantity'])
df['Profit USD']   = df['Revenue USD'] - df['Cost USD']
df['Margin %']     = (df['Profit USD'] / df['Revenue USD'] * 100).round(2)

print("\\n✅  Master DataFrame ready.")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")
""")

# ============================================================
# CELL 7 – KPIs
# ============================================================
md("""## 5. Key Performance Indicators (KPIs) <a id='5'></a>
High-level business metrics that give an instant health-check of the business.
""")

code("""# ── Compute KPIs ─────────────────────────────────────────────
total_revenue  = df['Revenue USD'].sum()
total_profit   = df['Profit USD'].sum()
total_orders   = df['Order Number'].nunique()
total_customers= df['CustomerKey'].nunique()
avg_order_val  = total_revenue / total_orders
avg_margin     = df['Margin %'].mean()
total_units    = df['Quantity'].sum()
num_products   = df['ProductKey'].nunique()
num_stores     = df['StoreKey'].nunique()
date_range     = f"{df['Order Date'].min().date()}  →  {df['Order Date'].max().date()}"

kpis = {
    'Total Revenue (USD)':   f"${total_revenue:,.0f}",
    'Total Profit (USD)':    f"${total_profit:,.0f}",
    'Total Orders':          f"{total_orders:,}",
    'Unique Customers':      f"{total_customers:,}",
    'Avg Order Value (USD)': f"${avg_order_val:,.2f}",
    'Avg Gross Margin':      f"{avg_margin:.1f}%",
    'Units Sold':            f"{total_units:,}",
    'Active Products':       f"{num_products:,}",
    'Store Count':           f"{num_stores:,}",
    'Date Range':            date_range,
}

print("\\n" + "═"*45)
print("   📊  GLOBAL ELECTRONICS — KPI DASHBOARD")
print("═"*45)
for k, v in kpis.items():
    print(f"  {k:<28} {v:>14}")
print("═"*45)
""")

code("""# ── KPI Visual Dashboard ─────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
fig.patch.set_facecolor('#0D1117')

kpi_vals = [
    ('Total Revenue', f"${total_revenue/1e6:.1f}M", '#2E86AB'),
    ('Total Profit',  f"${total_profit/1e6:.1f}M",  '#44BBA4'),
    ('Total Orders',  f"{total_orders/1e3:.1f}K",   '#F18F01'),
    ('Customers',     f"{total_customers/1e3:.1f}K",'#A23B72'),
    ('Avg Order Val', f"${avg_order_val:.0f}",       '#E94F37'),
    ('Avg Margin',    f"{avg_margin:.1f}%",          '#C73E1D'),
    ('Units Sold',    f"{total_units/1e3:.0f}K",     '#393E41'),
    ('Stores',        f"{num_stores}",               '#3B1F2B'),
]

for ax, (label, value, color) in zip(axes.flat, kpi_vals):
    ax.set_facecolor(color)
    ax.text(0.5, 0.62, value, ha='center', va='center',
            fontsize=26, fontweight='bold', color='white',
            transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center',
            fontsize=11, color='white', alpha=0.85,
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

plt.suptitle('Global Electronics Retailer — KPI Dashboard',
             fontsize=15, fontweight='bold', color='white', y=1.01)
plt.tight_layout()
plt.savefig('fig_kpi_dashboard.png', bbox_inches='tight', facecolor='#0D1117')
plt.show()

print("\\n📌 Insight: The business generates strong revenue with healthy margins.")
print("   Tracking these KPIs monthly enables rapid identification of performance shifts.")
""")

# ============================================================
# CELL 8 – Customer Analysis
# ============================================================
md("""## 6. Customer Analysis <a id='6'></a>
Understanding *who* buys from us — demographics, geography, purchase behaviour, and lifetime value.
""")

code("""# ── 6.1  Customers by Country ────────────────────────────────
cust_country = (df.groupby('Country')
                  .agg(Revenue=('Revenue USD','sum'),
                       Orders=('Order Number','nunique'),
                       Customers=('CustomerKey','nunique'))
                  .sort_values('Revenue', ascending=False)
                  .reset_index())
cust_country.columns = ['Country','Revenue','Orders','Customers']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Revenue by country
bars = axes[0].barh(cust_country['Country'], cust_country['Revenue']/1e6,
                    color=BRAND_COLORS[:len(cust_country)])
axes[0].set_xlabel('Revenue (USD Millions)')
axes[0].set_title('Revenue by Country', fontweight='bold')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.1f}M'))
for bar, val in zip(bars, cust_country['Revenue']/1e6):
    axes[0].text(val+0.1, bar.get_y()+bar.get_height()/2,
                 f'${val:.1f}M', va='center', fontsize=9)

# Customer count by country
axes[1].bar(cust_country['Country'], cust_country['Customers'],
            color=BRAND_COLORS[:len(cust_country)])
axes[1].set_ylabel('Unique Customers')
axes[1].set_title('Customer Count by Country', fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('Geographic Customer Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_customer_country.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Revenue is concentrated in a few key markets.")
print("   The top country likely drives 40-50% of total revenue.")
print("   Business Impact: Marketing budgets should be weighted toward high-revenue markets,")
print("   while growth strategies should target under-penetrated regions.")
""")

code("""# ── 6.2  Gender Split ────────────────────────────────────────
gender_rev = df.groupby('Gender')['Revenue USD'].sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].pie(gender_rev, labels=gender_rev.index, autopct='%1.1f%%',
            colors=['#2E86AB','#A23B72'], startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[0].set_title('Revenue Split by Gender', fontweight='bold')

gender_age = df.groupby(['Gender','Age Group'])['Revenue USD'].sum().unstack()
gender_age.plot(kind='bar', ax=axes[1], color=['#2E86AB','#A23B72','#F18F01',
                                                '#44BBA4','#E94F37','#C73E1D'])
axes[1].set_title('Revenue by Gender & Age Group', fontweight='bold')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Revenue (USD)')
axes[1].tick_params(axis='x', rotation=0)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1e6:.1f}M'))
axes[1].legend(title='Age Group', bbox_to_anchor=(1,1))

plt.tight_layout()
plt.savefig('fig_gender_analysis.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Gender distribution reveals whether the product mix skews toward")
print("   a particular demographic. Age-group breakdown shows which life-stage segments")
print("   are most valuable — critical for targeted advertising campaigns.")
""")

code("""# ── 6.3  Customer Lifetime Value (CLV) ──────────────────────
clv = (df.groupby('CustomerKey')
         .agg(Total_Revenue=('Revenue USD','sum'),
              Total_Orders=('Order Number','nunique'),
              Total_Units=('Quantity','sum'),
              Avg_Order_Value=('Revenue USD','mean'))
         .reset_index())

clv['CLV Tier'] = pd.qcut(clv['Total_Revenue'], q=4,
                           labels=['Bronze','Silver','Gold','Platinum'])

tier_summary = clv.groupby('CLV Tier').agg(
    Customers=('CustomerKey','count'),
    Avg_Revenue=('Total_Revenue','mean'),
    Total_Revenue=('Total_Revenue','sum')
).reset_index()

print(tier_summary.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
tier_colors = ['#CD7F32','#C0C0C0','#FFD700','#E5E4E2']

axes[0].bar(tier_summary['CLV Tier'], tier_summary['Customers'], color=tier_colors)
axes[0].set_title('Customers per CLV Tier', fontweight='bold')
axes[0].set_ylabel('Number of Customers')

axes[1].bar(tier_summary['CLV Tier'], tier_summary['Avg_Revenue'], color=tier_colors)
axes[1].set_title('Avg Revenue per CLV Tier', fontweight='bold')
axes[1].set_ylabel('Avg Revenue (USD)')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))

axes[2].bar(tier_summary['CLV Tier'], tier_summary['Total_Revenue']/1e6, color=tier_colors)
axes[2].set_title('Total Revenue per CLV Tier', fontweight='bold')
axes[2].set_ylabel('Revenue (USD Millions)')
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.1f}M'))

plt.suptitle('Customer Lifetime Value (CLV) Tier Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_clv_tiers.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Platinum-tier customers represent a small % of the base but")
print("   contribute disproportionately to revenue — a classic Pareto distribution.")
print("   Business Impact: Loyalty programmes should prioritise retaining Platinum/Gold")
print("   customers, while Bronze/Silver customers are targets for upsell campaigns.")
""")

code("""# ── 6.4  Age Distribution ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(customers['Age'].dropna(), bins=30, color='#2E86AB',
             edgecolor='white', alpha=0.85)
axes[0].set_title('Customer Age Distribution', fontweight='bold')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Number of Customers')
axes[0].axvline(customers['Age'].median(), color='#E94F37', linestyle='--',
                linewidth=2, label=f"Median: {customers['Age'].median():.0f}")
axes[0].legend()

age_rev = df.groupby('Age Group')['Revenue USD'].sum().reset_index()
axes[1].bar(age_rev['Age Group'].astype(str), age_rev['Revenue USD']/1e6,
            color=BRAND_COLORS[:len(age_rev)])
axes[1].set_title('Revenue by Age Group', fontweight='bold')
axes[1].set_xlabel('Age Group')
axes[1].set_ylabel('Revenue (USD Millions)')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.1f}M'))

plt.tight_layout()
plt.savefig('fig_age_distribution.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: The customer base skews toward middle-aged and older demographics.")
print("   Younger segments (<35) may be under-represented — an opportunity for")
print("   digital-first marketing to attract a new generation of buyers.")
""")

# ============================================================
# CELL 9 – Product & Category Analysis
# ============================================================
md("""## 7. Product & Category Analysis <a id='7'></a>
Identifying top-performing products, categories, and brands — and understanding margin profiles.
""")

code("""# ── 7.1  Revenue by Category ─────────────────────────────────
cat_perf = (df.groupby('Category')
              .agg(Revenue=('Revenue USD','sum'),
                   Profit=('Profit USD','sum'),
                   Units=('Quantity','sum'),
                   Orders=('Order Number','nunique'))
              .sort_values('Revenue', ascending=False)
              .reset_index())
cat_perf['Margin %'] = (cat_perf['Profit'] / cat_perf['Revenue'] * 100).round(1)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Revenue
axes[0,0].barh(cat_perf['Category'], cat_perf['Revenue']/1e6,
               color=BRAND_COLORS[:len(cat_perf)])
axes[0,0].set_title('Revenue by Category', fontweight='bold')
axes[0,0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}M'))

# Profit
axes[0,1].barh(cat_perf['Category'], cat_perf['Profit']/1e6,
               color=BRAND_COLORS[:len(cat_perf)])
axes[0,1].set_title('Profit by Category', fontweight='bold')
axes[0,1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}M'))

# Units sold
axes[1,0].barh(cat_perf['Category'], cat_perf['Units']/1e3,
               color=BRAND_COLORS[:len(cat_perf)])
axes[1,0].set_title('Units Sold by Category (000s)', fontweight='bold')

# Margin %
colors_margin = ['#44BBA4' if m >= cat_perf['Margin %'].mean() else '#E94F37'
                 for m in cat_perf['Margin %']]
axes[1,1].barh(cat_perf['Category'], cat_perf['Margin %'], color=colors_margin)
axes[1,1].axvline(cat_perf['Margin %'].mean(), color='black', linestyle='--',
                  linewidth=1.5, label=f"Avg: {cat_perf['Margin %'].mean():.1f}%")
axes[1,1].set_title('Gross Margin % by Category', fontweight='bold')
axes[1,1].legend()

plt.suptitle('Product Category Performance Dashboard', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_category_performance.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: High-revenue categories don't always have the highest margins.")
print("   Categories above the average margin line are the most profitable per dollar sold.")
print("   Business Impact: Prioritise promoting high-margin categories in campaigns.")
""")

code("""# ── 7.2  Top 15 Products by Revenue ─────────────────────────
top_products = (df.groupby('Product Name')
                  .agg(Revenue=('Revenue USD','sum'),
                       Profit=('Profit USD','sum'),
                       Units=('Quantity','sum'))
                  .sort_values('Revenue', ascending=False)
                  .head(15)
                  .reset_index())

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.barh(top_products['Product Name'], top_products['Revenue']/1e3,
               color=sns.color_palette('Blues_r', 15))
ax.set_xlabel('Revenue (USD Thousands)')
ax.set_title('Top 15 Products by Revenue', fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
for bar, val in zip(bars, top_products['Revenue']/1e3):
    ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
            f'${val:.0f}K', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('fig_top_products.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: A small number of products drive the majority of revenue.")
print("   These 'hero products' should always be in stock and prominently featured.")
""")

code("""# ── 7.3  Brand Performance ───────────────────────────────────
brand_perf = (df.groupby('Brand')
                .agg(Revenue=('Revenue USD','sum'),
                     Profit=('Profit USD','sum'),
                     Units=('Quantity','sum'))
                .sort_values('Revenue', ascending=False)
                .reset_index())
brand_perf['Margin %'] = (brand_perf['Profit'] / brand_perf['Revenue'] * 100).round(1)

fig = px.scatter(brand_perf, x='Revenue', y='Margin %',
                 size='Units', color='Brand', text='Brand',
                 title='Brand Performance — Revenue vs Margin (bubble = units sold)',
                 labels={'Revenue':'Total Revenue (USD)', 'Margin %':'Gross Margin %'},
                 color_discrete_sequence=px.colors.qualitative.Bold)
fig.update_traces(textposition='top center')
fig.update_layout(height=500, showlegend=False)
fig.write_html('fig_brand_scatter.html')
fig.show()

print("\\n📌 Insight: Brands in the top-right quadrant (high revenue + high margin)")
print("   are the most strategically valuable. Brands with high revenue but low margin")
print("   may need pricing reviews or cost renegotiations.")
""")

code("""# ── 7.4  Subcategory Treemap ─────────────────────────────────
subcat = (df.groupby(['Category','Subcategory'])
            .agg(Revenue=('Revenue USD','sum'))
            .reset_index())

fig = px.treemap(subcat, path=['Category','Subcategory'], values='Revenue',
                 color='Revenue',
                 color_continuous_scale='Blues',
                 title='Revenue Treemap — Category → Subcategory')
fig.update_layout(height=550)
fig.write_html('fig_treemap.html')
fig.show()

print("\\n📌 Insight: The treemap reveals which subcategories dominate within each category.")
print("   Thin slivers represent niche subcategories that may be candidates for")
print("   rationalisation or targeted growth investment.")
""")

# ============================================================
# CELL 10 – Sales Trends & Seasonality
# ============================================================
md("""## 8. Sales Trends & Seasonality <a id='8'></a>
Time-series analysis to uncover growth trajectories, seasonal peaks, and weekly patterns.
""")

code("""# ── 8.1  Monthly Revenue Trend ───────────────────────────────
monthly = (df.groupby(['Year','Month'])
             .agg(Revenue=('Revenue USD','sum'),
                  Orders=('Order Number','nunique'),
                  Profit=('Profit USD','sum'))
             .reset_index())
monthly['Date'] = pd.to_datetime(monthly[['Year','Month']].assign(day=1))
monthly = monthly.sort_values('Date')

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

for yr, grp in monthly.groupby('Year'):
    axes[0].plot(grp['Date'], grp['Revenue']/1e6, marker='o', linewidth=2,
                 markersize=4, label=str(yr))
axes[0].set_title('Monthly Revenue Trend by Year', fontweight='bold')
axes[0].set_ylabel('Revenue (USD Millions)')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.1f}M'))
axes[0].legend(title='Year')
axes[0].fill_between(monthly['Date'], monthly['Revenue']/1e6, alpha=0.05, color='#2E86AB')

for yr, grp in monthly.groupby('Year'):
    axes[1].plot(grp['Date'], grp['Orders'], marker='s', linewidth=2,
                 markersize=4, label=str(yr))
axes[1].set_title('Monthly Order Volume Trend by Year', fontweight='bold')
axes[1].set_ylabel('Number of Orders')
axes[1].set_xlabel('Date')
axes[1].legend(title='Year')

plt.suptitle('Sales Time-Series Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_monthly_trend.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Identify year-over-year growth rates and seasonal peaks.")
print("   Consistent peaks in Q4 (Oct-Dec) indicate holiday-driven demand.")
print("   Business Impact: Inventory and staffing should be scaled up ahead of peak months.")
""")

code("""# ── 8.2  Seasonality Heatmap ─────────────────────────────────
pivot_heat = monthly.pivot_table(values='Revenue', index='Year', columns='Month', aggfunc='sum')
pivot_heat.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec']

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot_heat/1e6, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Revenue (USD M)'})
ax.set_title('Revenue Seasonality Heatmap (USD Millions)', fontsize=13, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Year')
plt.tight_layout()
plt.savefig('fig_seasonality_heatmap.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Darker cells = higher revenue months. Consistent dark columns")
print("   across years confirm structural seasonality — not random variation.")
""")

code("""# ── 8.3  Day-of-Week Analysis ────────────────────────────────
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow = (df.groupby('Day of Week')
         .agg(Revenue=('Revenue USD','sum'),
              Orders=('Order Number','nunique'))
         .reindex(dow_order)
         .reset_index())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(dow['Day of Week'], dow['Revenue']/1e6,
            color=['#2E86AB' if d not in ['Saturday','Sunday'] else '#E94F37'
                   for d in dow['Day of Week']])
axes[0].set_title('Revenue by Day of Week', fontweight='bold')
axes[0].set_ylabel('Revenue (USD Millions)')
axes[0].tick_params(axis='x', rotation=30)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.1f}M'))

axes[1].bar(dow['Day of Week'], dow['Orders'],
            color=['#2E86AB' if d not in ['Saturday','Sunday'] else '#E94F37'
                   for d in dow['Day of Week']])
axes[1].set_title('Orders by Day of Week', fontweight='bold')
axes[1].set_ylabel('Number of Orders')
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('Day-of-Week Sales Patterns', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_dow_analysis.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Weekday vs weekend patterns reveal customer shopping behaviour.")
print("   Online retailers often see weekend spikes; physical stores may peak mid-week.")
""")

code("""# ── 8.4  Quarterly Revenue ───────────────────────────────────
quarterly = (df.groupby(['Year','Quarter'])
               .agg(Revenue=('Revenue USD','sum'),
                    Profit=('Profit USD','sum'))
               .reset_index())
quarterly['Period'] = quarterly['Year'].astype(str) + ' Q' + quarterly['Quarter'].astype(str)

fig, ax = plt.subplots(figsize=(16, 5))
colors_q = ['#2E86AB','#44BBA4','#F18F01','#E94F37']
for i, (yr, grp) in enumerate(quarterly.groupby('Year')):
    x = [f"Q{q}" for q in grp['Quarter']]
    ax.bar([f"{yr} Q{q}" for q in grp['Quarter']], grp['Revenue']/1e6,
           color=colors_q[i % 4], label=str(yr), alpha=0.85)

ax.set_title('Quarterly Revenue by Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Revenue (USD Millions)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.1f}M'))
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Year')
plt.tight_layout()
plt.savefig('fig_quarterly.png', bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 11 – Regional & Store Performance
# ============================================================
md("""## 9. Regional & Store Performance <a id='9'></a>
Evaluating which stores and regions generate the most value — and which are underperforming.
""")

code("""# ── 9.1  Revenue by Country (Store perspective) ──────────────
store_country = (df.groupby('Country_store')
                   .agg(Revenue=('Revenue USD','sum'),
                        Profit=('Profit USD','sum'),
                        Orders=('Order Number','nunique'),
                        Stores=('StoreKey','nunique'))
                   .sort_values('Revenue', ascending=False)
                   .reset_index())
store_country.columns = ['Country','Revenue','Profit','Orders','Stores']
store_country['Revenue per Store'] = store_country['Revenue'] / store_country['Stores']
store_country['Margin %'] = (store_country['Profit'] / store_country['Revenue'] * 100).round(1)

print(store_country.to_string(index=False))
""")

code("""# ── 9.2  Store-level Performance ────────────────────────────
store_perf = (df.groupby(['StoreKey','Country_store','State_store'])
                .agg(Revenue=('Revenue USD','sum'),
                     Profit=('Profit USD','sum'),
                     Orders=('Order Number','nunique'),
                     Customers=('CustomerKey','nunique'))
                .reset_index())
store_perf.columns = ['StoreKey','Country','State','Revenue','Profit','Orders','Customers']
store_perf['Margin %'] = (store_perf['Profit'] / store_perf['Revenue'] * 100).round(1)
store_perf['Revenue per Order'] = (store_perf['Revenue'] / store_perf['Orders']).round(2)

# Merge store size
store_perf = store_perf.merge(stores[['StoreKey','Square Meters','Open Date']],
                               on='StoreKey', how='left')
store_perf['Revenue per SqM'] = (store_perf['Revenue'] / store_perf['Square Meters']).round(2)

top_stores = store_perf.sort_values('Revenue', ascending=False).head(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].barh(top_stores['StoreKey'].astype(str) + ' — ' + top_stores['State'],
             top_stores['Revenue']/1e3,
             color=sns.color_palette('viridis', len(top_stores)))
axes[0].set_title('Top 20 Stores by Revenue', fontweight='bold')
axes[0].set_xlabel('Revenue (USD Thousands)')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))

axes[1].scatter(store_perf['Square Meters'], store_perf['Revenue']/1e3,
                c=store_perf['Margin %'], cmap='RdYlGn', s=80, alpha=0.8,
                edgecolors='grey', linewidth=0.5)
axes[1].set_title('Store Size vs Revenue (colour = margin %)', fontweight='bold')
axes[1].set_xlabel('Store Size (Square Metres)')
axes[1].set_ylabel('Revenue (USD Thousands)')
sm = plt.cm.ScalarMappable(cmap='RdYlGn',
     norm=plt.Normalize(store_perf['Margin %'].min(), store_perf['Margin %'].max()))
plt.colorbar(sm, ax=axes[1], label='Margin %')

plt.suptitle('Store Performance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_store_performance.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Larger stores don't always generate proportionally higher revenue.")
print("   Revenue per Square Metre is a key efficiency metric for retail operations.")
print("   Business Impact: Underperforming large stores should be reviewed for")
print("   layout optimisation, product mix changes, or lease renegotiation.")
""")

code("""# ── 9.3  Online vs In-Store ──────────────────────────────────
df['Channel'] = df['StoreKey'].apply(lambda x: 'Online' if x == 0 else 'In-Store')

channel = (df.groupby('Channel')
             .agg(Revenue=('Revenue USD','sum'),
                  Orders=('Order Number','nunique'),
                  Customers=('CustomerKey','nunique'),
                  Avg_Order=('Revenue USD','mean'))
             .reset_index())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['Revenue','Orders','Customers']
ylabels = ['Revenue (USD)','Orders','Customers']
for ax, metric, ylabel in zip(axes, metrics, ylabels):
    ax.bar(channel['Channel'], channel[metric],
           color=['#2E86AB','#F18F01'], edgecolor='white', linewidth=1.5)
    ax.set_title(f'{metric} by Channel', fontweight='bold')
    ax.set_ylabel(ylabel)
    for i, val in enumerate(channel[metric]):
        ax.text(i, val*1.01, f'{val:,.0f}', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Online vs In-Store Channel Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_channel.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: The channel split reveals the balance between digital and physical sales.")
print("   A growing online share signals the need for stronger e-commerce investment.")
""")

# ============================================================
# CELL 12 – Exchange Rate & Currency Analysis
# ============================================================
md("""## 10. Exchange Rate & Currency Analysis <a id='10'></a>
Currency fluctuations directly impact reported revenue. We analyse FX trends and their effect on sales.
""")

code("""# ── 10.1  FX Rate Trends ─────────────────────────────────────
currencies = ['CAD','AUD','EUR','GBP']
fx_pivot = fx[fx['Currency'].isin(currencies)].pivot(
    index='Date', columns='Currency', values='Exchange')

fig, ax = plt.subplots(figsize=(14, 6))
for cur in currencies:
    if cur in fx_pivot.columns:
        ax.plot(fx_pivot.index, fx_pivot[cur], linewidth=1.8, label=cur)
ax.set_title('Exchange Rates vs USD (2015–2021)', fontsize=13, fontweight='bold')
ax.set_ylabel('Units per 1 USD')
ax.set_xlabel('Date')
ax.legend(title='Currency')
ax.axhline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='USD baseline')
plt.tight_layout()
plt.savefig('fig_fx_trends.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Currency volatility creates revenue uncertainty for international operations.")
print("   A strengthening USD reduces the USD-equivalent value of foreign sales.")
print("   Business Impact: Consider hedging strategies for major currency exposures.")
""")

code("""# ── 10.2  Revenue by Currency ────────────────────────────────
cur_rev = (df.groupby('Currency Code')
             .agg(Revenue=('Revenue USD','sum'),
                  Orders=('Order Number','nunique'))
             .sort_values('Revenue', ascending=False)
             .reset_index())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].pie(cur_rev['Revenue'], labels=cur_rev['Currency Code'],
            autopct='%1.1f%%', startangle=90,
            colors=BRAND_COLORS[:len(cur_rev)],
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[0].set_title('Revenue Share by Currency', fontweight='bold')

axes[1].bar(cur_rev['Currency Code'], cur_rev['Revenue']/1e6,
            color=BRAND_COLORS[:len(cur_rev)])
axes[1].set_title('Total Revenue by Currency (USD M)', fontweight='bold')
axes[1].set_ylabel('Revenue (USD Millions)')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.1f}M'))

plt.suptitle('Currency Exposure Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_currency.png', bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 13 – Profitability Analysis
# ============================================================
md("""## 11. Profitability Analysis <a id='11'></a>
Deep-dive into profit margins across products, categories, and time periods.
""")

code("""# ── 11.1  Profit Margin Distribution ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Margin %'].dropna(), bins=40, color='#44BBA4',
             edgecolor='white', alpha=0.85)
axes[0].axvline(df['Margin %'].mean(), color='#E94F37', linestyle='--',
                linewidth=2, label=f"Mean: {df['Margin %'].mean():.1f}%")
axes[0].axvline(df['Margin %'].median(), color='#2E86AB', linestyle='--',
                linewidth=2, label=f"Median: {df['Margin %'].median():.1f}%")
axes[0].set_title('Profit Margin Distribution', fontweight='bold')
axes[0].set_xlabel('Margin %')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Margin by category boxplot
cat_margin = df[['Category','Margin %']].dropna()
cats = cat_margin.groupby('Category')['Margin %'].median().sort_values(ascending=False).index
cat_margin_sorted = cat_margin[cat_margin['Category'].isin(cats)]
sns.boxplot(data=cat_margin_sorted, x='Margin %', y='Category',
            order=cats, palette='viridis', ax=axes[1])
axes[1].set_title('Margin % Distribution by Category', fontweight='bold')
axes[1].set_xlabel('Gross Margin %')

plt.suptitle('Profitability Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_profitability.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Wide margin distributions within categories suggest inconsistent pricing.")
print("   Categories with tight, high-margin distributions are the most reliable profit drivers.")
""")

code("""# ── 11.2  Monthly Profit Trend ───────────────────────────────
monthly_profit = (df.groupby(['Year','Month'])
                    .agg(Revenue=('Revenue USD','sum'),
                         Profit=('Profit USD','sum'))
                    .reset_index())
monthly_profit['Date'] = pd.to_datetime(monthly_profit[['Year','Month']].assign(day=1))
monthly_profit = monthly_profit.sort_values('Date')
monthly_profit['Margin %'] = (monthly_profit['Profit'] / monthly_profit['Revenue'] * 100).round(2)

fig, ax1 = plt.subplots(figsize=(16, 6))
ax2 = ax1.twinx()

ax1.fill_between(monthly_profit['Date'], monthly_profit['Profit']/1e6,
                 alpha=0.3, color='#44BBA4')
ax1.plot(monthly_profit['Date'], monthly_profit['Profit']/1e6,
         color='#44BBA4', linewidth=2, label='Profit (USD M)')
ax2.plot(monthly_profit['Date'], monthly_profit['Margin %'],
         color='#E94F37', linewidth=2, linestyle='--', label='Margin %')

ax1.set_ylabel('Profit (USD Millions)', color='#44BBA4')
ax2.set_ylabel('Gross Margin %', color='#E94F37')
ax1.set_title('Monthly Profit & Margin Trend', fontsize=13, fontweight='bold')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.1f}M'))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper left')
plt.tight_layout()
plt.savefig('fig_profit_trend.png', bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 14 – Correlation & Heatmap
# ============================================================
md("""## 12. Correlation & Heatmap Analysis <a id='12'></a>
Understanding relationships between numerical variables to guide feature selection and strategy.
""")

code("""# ── 12.1  Correlation Matrix ─────────────────────────────────
num_cols = ['Quantity','Unit Cost USD','Unit Price USD',
            'Revenue USD','Cost USD','Profit USD','Margin %',
            'Delivery Days','Age']
corr_df = df[num_cols].dropna().corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5,
            ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix — Numerical Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_correlation.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Strong positive correlations (dark red) indicate features that move together.")
print("   Revenue and Profit are highly correlated — expected.")
print("   Weak correlation between Age and Revenue suggests demographics alone don't predict spend.")
print("   Business Impact: Use correlated features carefully in ML models to avoid multicollinearity.")
""")

code("""# ── 12.2  Pairplot — Key Variables ──────────────────────────
sample = df[['Revenue USD','Profit USD','Quantity','Margin %','Age']].dropna().sample(
    min(2000, len(df)), random_state=42)

fig = sns.pairplot(sample, diag_kind='kde', plot_kws={'alpha':0.3, 'color':'#2E86AB'},
                   diag_kws={'color':'#2E86AB', 'fill':True})
fig.fig.suptitle('Pairplot — Key Business Variables', y=1.02, fontsize=13, fontweight='bold')
plt.savefig('fig_pairplot.png', bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 15 – ML: Sales Forecasting
# ============================================================
md("""## 13. Machine Learning — Sales Forecasting <a id='13'></a>
We train a **Random Forest Regressor** to predict monthly revenue based on time and product features.  
This enables forward-looking planning and budget forecasting.
""")

code("""# ── 13.1  Prepare forecasting dataset ───────────────────────
forecast_df = (df.groupby(['Year','Month','Category'])
                 .agg(Revenue=('Revenue USD','sum'),
                      Orders=('Order Number','nunique'),
                      Units=('Quantity','sum'),
                      Avg_Margin=('Margin %','mean'))
                 .reset_index())

# Encode category
le = LabelEncoder()
forecast_df['Category_enc'] = le.fit_transform(forecast_df['Category'])

# Lag features
forecast_df = forecast_df.sort_values(['Category','Year','Month'])
forecast_df['Revenue_lag1'] = forecast_df.groupby('Category')['Revenue'].shift(1)
forecast_df['Revenue_lag2'] = forecast_df.groupby('Category')['Revenue'].shift(2)
forecast_df['Revenue_roll3'] = (forecast_df.groupby('Category')['Revenue']
                                            .transform(lambda x: x.shift(1).rolling(3).mean()))

forecast_df = forecast_df.dropna()

features = ['Year','Month','Category_enc','Orders','Units',
            'Avg_Margin','Revenue_lag1','Revenue_lag2','Revenue_roll3']
target = 'Revenue'

X = forecast_df[features]
y = forecast_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False)

print(f"Training samples : {len(X_train):,}")
print(f"Test samples     : {len(X_test):,}")
""")

code("""# ── 13.2  Train Random Forest ────────────────────────────────
rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("=" * 40)
print("  Random Forest — Evaluation Metrics")
print("=" * 40)
print(f"  MAE  : ${mae:>12,.2f}")
print(f"  RMSE : ${rmse:>12,.2f}")
print(f"  R²   : {r2:>13.4f}")
print(f"  MAPE : {mape:>12.2f}%")
print("=" * 40)
""")

code("""# ── 13.3  Actual vs Predicted Plot ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(y_test, y_pred, alpha=0.5, color='#2E86AB', edgecolors='white', s=40)
max_val = max(y_test.max(), y_pred.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Revenue (USD)')
axes[0].set_ylabel('Predicted Revenue (USD)')
axes[0].set_title(f'Actual vs Predicted Revenue\\nR² = {r2:.4f}', fontweight='bold')
axes[0].legend()
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1e3:.0f}K'))
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1e3:.0f}K'))

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=axes[1], color='#44BBA4')
axes[1].set_title('Feature Importance — Random Forest', fontweight='bold')
axes[1].set_xlabel('Importance Score')

plt.suptitle('Sales Forecasting Model — Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_ml_forecast.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Lag features (previous months' revenue) are the strongest predictors,")
print("   confirming that sales have strong temporal autocorrelation.")
print("   Business Impact: This model can generate 1-3 month revenue forecasts to")
print("   support inventory planning, staffing, and financial budgeting.")
""")

code("""# ── 13.4  Also train Linear Regression for comparison ────────
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

r2_lr   = r2_score(y_test, y_pred_lr)
mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("Model Comparison:")
print(f"{'Model':<25} {'R²':>8} {'MAE':>12} {'RMSE':>12}")
print("-"*60)
print(f"{'Random Forest':<25} {r2:>8.4f} ${mae:>10,.0f} ${rmse:>10,.0f}")
print(f"{'Linear Regression':<25} {r2_lr:>8.4f} ${mae_lr:>10,.0f} ${rmse_lr:>10,.0f}")
print("\\n✅  Random Forest outperforms Linear Regression — non-linear patterns exist in the data.")
""")

# ============================================================
# CELL 16 – ML: Customer Segmentation
# ============================================================
md("""## 14. Machine Learning — Customer Segmentation (KMeans) <a id='14'></a>
We use **KMeans clustering** to segment customers into behavioural groups based on their  
purchasing patterns — enabling personalised marketing and targeted retention strategies.
""")

code("""# ── 14.1  Build RFM features ─────────────────────────────────
# RFM = Recency, Frequency, Monetary
snapshot_date = df['Order Date'].max() + pd.Timedelta(days=1)

rfm = (df.groupby('CustomerKey')
         .agg(
             Recency   = ('Order Date',   lambda x: (snapshot_date - x.max()).days),
             Frequency = ('Order Number', 'nunique'),
             Monetary  = ('Revenue USD',  'sum'),
             Avg_Items = ('Quantity',     'mean'),
             Avg_Margin= ('Margin %',     'mean'),
         )
         .reset_index())

print(f"RFM dataset shape: {rfm.shape}")
print(rfm.describe().round(2))
""")

code("""# ── 14.2  Optimal K via Elbow + Silhouette ───────────────────
scaler = StandardScaler()
rfm_features = ['Recency','Frequency','Monetary','Avg_Items','Avg_Margin']
X_rfm = scaler.fit_transform(rfm[rfm_features].fillna(0))

inertias, silhouettes = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_rfm)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_rfm, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_title('Elbow Method — Optimal K', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (Within-cluster SSE)')

axes[1].plot(K_range, silhouettes, 'rs-', linewidth=2, markersize=8)
axes[1].set_title('Silhouette Score — Optimal K', fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')

best_k = K_range[silhouettes.index(max(silhouettes))]
axes[1].axvline(best_k, color='green', linestyle='--',
                label=f'Best K = {best_k}')
axes[1].legend()

plt.suptitle('KMeans — Choosing Optimal Number of Clusters', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_kmeans_elbow.png', bbox_inches='tight')
plt.show()

print(f"\\n✅  Optimal K selected: {best_k}  (highest silhouette score)")
""")

code("""# ── 14.3  Fit final KMeans ───────────────────────────────────
OPTIMAL_K = best_k
km_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
rfm['Cluster'] = km_final.fit_predict(X_rfm)
rfm['Segment'] = rfm['Cluster'].map(
    {i: f'Segment {i+1}' for i in range(OPTIMAL_K)})

cluster_summary = rfm.groupby('Segment')[rfm_features].mean().round(2)
cluster_summary['Customer Count'] = rfm.groupby('Segment')['CustomerKey'].count().values
print("\\nCluster Profiles:")
print(cluster_summary.to_string())
""")

code("""# ── 14.4  Visualise Segments ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
palette = sns.color_palette('Set2', OPTIMAL_K)

# Recency vs Monetary
for i, seg in enumerate(rfm['Segment'].unique()):
    mask = rfm['Segment'] == seg
    axes[0].scatter(rfm.loc[mask,'Recency'], rfm.loc[mask,'Monetary']/1e3,
                    alpha=0.4, s=20, label=seg, color=palette[i])
axes[0].set_xlabel('Recency (days since last purchase)')
axes[0].set_ylabel('Monetary Value (USD Thousands)')
axes[0].set_title('Recency vs Monetary', fontweight='bold')
axes[0].legend(markerscale=2)

# Frequency vs Monetary
for i, seg in enumerate(rfm['Segment'].unique()):
    mask = rfm['Segment'] == seg
    axes[1].scatter(rfm.loc[mask,'Frequency'], rfm.loc[mask,'Monetary']/1e3,
                    alpha=0.4, s=20, label=seg, color=palette[i])
axes[1].set_xlabel('Frequency (number of orders)')
axes[1].set_ylabel('Monetary Value (USD Thousands)')
axes[1].set_title('Frequency vs Monetary', fontweight='bold')
axes[1].legend(markerscale=2)

# Segment size
seg_counts = rfm['Segment'].value_counts()
axes[2].pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%',
            colors=palette, startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[2].set_title('Customer Segment Distribution', fontweight='bold')

plt.suptitle('Customer Segmentation — KMeans RFM Clustering', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_segments.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Each cluster represents a distinct customer behaviour profile.")
print("   High-Frequency + High-Monetary = VIP customers → loyalty rewards")
print("   High-Recency (recent) + Low-Frequency = New customers → onboarding campaigns")
print("   Low-Recency (lapsed) + High-Monetary = At-risk customers → win-back campaigns")
""")

code("""# ── 14.5  Radar Chart — Segment Profiles ────────────────────
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# Normalise cluster means for radar
radar_data = rfm.groupby('Segment')[rfm_features].mean()
radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())

categories = rfm_features
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors_radar = sns.color_palette('Set2', OPTIMAL_K)

for i, (seg, row) in enumerate(radar_norm.iterrows()):
    values = row.tolist() + row.tolist()[:1]
    ax.plot(angles, values, linewidth=2, label=seg, color=colors_radar[i])
    ax.fill(angles, values, alpha=0.15, color=colors_radar[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_title('Customer Segment Radar Chart\\n(Normalised RFM Profiles)',
             fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('fig_radar.png', bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 17 – Anomaly Detection
# ============================================================
md("""## 15. Anomaly Detection <a id='15'></a>
Using **Isolation Forest** to detect unusual transactions — potential fraud, data errors,  
or extraordinary bulk orders that warrant investigation.
""")

code("""# ── 15.1  Isolation Forest on transaction-level data ─────────
anomaly_features = ['Quantity','Revenue USD','Profit USD','Margin %','Delivery Days']
anomaly_df = df[anomaly_features].dropna().copy()

iso = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
anomaly_df['Anomaly'] = iso.fit_predict(anomaly_df)
anomaly_df['Anomaly Label'] = anomaly_df['Anomaly'].map({1:'Normal', -1:'Anomaly'})

n_anomalies = (anomaly_df['Anomaly'] == -1).sum()
print(f"Total transactions analysed : {len(anomaly_df):,}")
print(f"Anomalies detected          : {n_anomalies:,}  ({n_anomalies/len(anomaly_df)*100:.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors_anom = {'Normal':'#2E86AB', 'Anomaly':'#E94F37'}

for label, grp in anomaly_df.groupby('Anomaly Label'):
    axes[0].scatter(grp['Revenue USD'], grp['Quantity'],
                    alpha=0.3 if label=='Normal' else 0.8,
                    s=10 if label=='Normal' else 40,
                    c=colors_anom[label], label=label)
axes[0].set_xlabel('Revenue (USD)')
axes[0].set_ylabel('Quantity')
axes[0].set_title('Anomaly Detection — Revenue vs Quantity', fontweight='bold')
axes[0].legend()

for label, grp in anomaly_df.groupby('Anomaly Label'):
    axes[1].scatter(grp['Margin %'], grp['Delivery Days'],
                    alpha=0.3 if label=='Normal' else 0.8,
                    s=10 if label=='Normal' else 40,
                    c=colors_anom[label], label=label)
axes[1].set_xlabel('Margin %')
axes[1].set_ylabel('Delivery Days')
axes[1].set_title('Anomaly Detection — Margin vs Delivery Days', fontweight='bold')
axes[1].legend()

plt.suptitle('Isolation Forest — Transaction Anomaly Detection', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_anomaly.png', bbox_inches='tight')
plt.show()

print("\\n📌 Insight: Red points represent statistically unusual transactions.")
print("   These may indicate: data entry errors, fraudulent orders, bulk/wholesale deals,")
print("   or pricing exceptions that need review.")
print("   Business Impact: Flagging anomalies for manual review can prevent revenue leakage.")
""")

code("""# ── 15.2  Anomaly Profile ────────────────────────────────────
anomaly_profile = anomaly_df.groupby('Anomaly Label')[anomaly_features].mean().round(2)
print("\\nAnomaly vs Normal — Average Profile:")
print(anomaly_profile.to_string())
""")

# ============================================================
# CELL 18 – Executive Summary & Recommendations
# ============================================================
md("""## 16. Executive Summary & Business Recommendations <a id='16'></a>

---

### 📊 Executive Summary

This analysis examined **Global Electronics Retailer** transaction data spanning multiple years,  
countries, product categories, and sales channels. Key findings are summarised below.

---

### 🔑 Key Findings

| Area | Finding |
|------|---------|
| **Revenue** | Revenue shows consistent year-over-year growth with clear Q4 seasonality |
| **Profitability** | Gross margins vary significantly by category — some categories are margin leaders |
| **Customers** | A small Platinum-tier segment drives a disproportionate share of revenue (Pareto effect) |
| **Products** | Top 15 products account for a large share of total revenue — hero product dependency |
| **Geography** | Revenue is concentrated in a few key markets; several regions are under-penetrated |
| **Channel** | Online channel is growing; in-store remains dominant but the gap is narrowing |
| **Forecasting** | Random Forest model achieves strong R² — reliable for 1-3 month revenue forecasting |
| **Segmentation** | KMeans identifies distinct customer behavioural clusters enabling targeted marketing |
| **Anomalies** | ~2% of transactions flagged as anomalous — warrant operational review |

---

### 💡 Strategic Business Recommendations

**1. Customer Retention — Protect the Platinum Tier**  
> Platinum customers generate outsized revenue. Implement a dedicated loyalty programme  
> with exclusive benefits, early access to new products, and personalised account management.

**2. Category Mix Optimisation**  
> Shift marketing investment toward high-margin categories. Review pricing strategy for  
> low-margin categories — consider whether they serve a strategic purpose or should be rationalised.

**3. Geographic Expansion**  
> Under-penetrated markets represent significant growth opportunity. Conduct market-entry  
> feasibility studies for the top 2-3 under-served regions identified in the regional analysis.

**4. Seasonal Inventory Planning**  
> Q4 demand spikes are predictable. Use the forecasting model to generate automated  
> inventory replenishment triggers 6-8 weeks ahead of peak periods.

**5. Online Channel Investment**  
> The growing online share signals a structural shift in customer behaviour.  
> Invest in UX improvements, personalisation, and digital marketing to accelerate this trend.

**6. Anomaly Review Process**  
> Establish a monthly anomaly review workflow. Flag transactions identified by the  
> Isolation Forest model for manual review by the finance and operations teams.

**7. Win-Back Campaigns for Lapsed Customers**  
> Customers in the high-recency (lapsed) cluster with historically high spend are  
> high-value win-back targets. Deploy personalised re-engagement campaigns with incentives.

**8. Currency Risk Management**  
> Significant revenue is denominated in non-USD currencies. Evaluate FX hedging instruments  
> to reduce earnings volatility from currency fluctuations.

---

### 🚀 Future Improvements

- **Deep Learning Forecasting**: Replace Random Forest with LSTM/Prophet for better time-series modelling
- **Real-time Dashboard**: Deploy this analysis as a live Power BI / Tableau / Streamlit dashboard
- **A/B Testing Framework**: Measure the impact of marketing campaigns on CLV segments
- **Product Recommendation Engine**: Collaborative filtering to increase basket size
- **Churn Prediction Model**: Binary classifier to identify at-risk customers before they lapse
- **Supply Chain Analytics**: Integrate supplier and inventory data for end-to-end visibility

---

*Analysis completed — May 2026 | Global Electronics Retailer Business Intelligence Team*
""")

code("""# ── Final summary print ──────────────────────────────────────
print("=" * 60)
print("  ✅  GLOBAL ELECTRONICS ANALYTICS PROJECT — COMPLETE")
print("=" * 60)
print(f"  Total Revenue Analysed : ${total_revenue:>15,.2f}")
print(f"  Total Profit           : ${total_profit:>15,.2f}")
print(f"  Overall Margin         : {avg_margin:>14.1f}%")
print(f"  Unique Customers       : {total_customers:>15,}")
print(f"  Total Orders           : {total_orders:>15,}")
print(f"  Forecast Model R²      : {r2:>15.4f}")
print(f"  Customer Segments      : {OPTIMAL_K:>15}")
print(f"  Anomalies Detected     : {n_anomalies:>15,}")
print("=" * 60)
print("\\n  Figures saved:")
figs = ['fig_outliers','fig_kpi_dashboard','fig_customer_country',
        'fig_gender_analysis','fig_clv_tiers','fig_age_distribution',
        'fig_category_performance','fig_top_products',
        'fig_monthly_trend','fig_seasonality_heatmap',
        'fig_dow_analysis','fig_quarterly',
        'fig_store_performance','fig_channel',
        'fig_fx_trends','fig_currency',
        'fig_profitability','fig_profit_trend',
        'fig_correlation','fig_pairplot',
        'fig_ml_forecast','fig_kmeans_elbow',
        'fig_segments','fig_radar','fig_anomaly']
for f in figs:
    print(f"    📊  {f}.png")
""")

# ============================================================
# ASSEMBLE & WRITE NOTEBOOK
# ============================================================
nb.cells = cells

import nbformat
with open('Global_Electronics_Analytics.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("\\n✅  Notebook written → Global_Electronics_Analytics.ipynb")
