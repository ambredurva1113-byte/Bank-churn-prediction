# ============================================================
# 🏦 BANK CUSTOMER CHURN — EDA & VISUALIZATION
# Tool    : Python IDLE
# Libraries: pandas | matplotlib | seaborn | numpy
# Dataset : Kaggle - Bank Customer Churn
# ============================================================
#
# HOW TO RUN IN IDLE:
#   1. Open this file in IDLE
#   2. Press F5 (or Run > Run Module)
#   3. Charts will open one by one — close each to see the next
#
# FOLDER SETUP (do this once):
#   • Place bank_churn.csv in the SAME folder as this .py file
#   • That's it! The script will create 'charts' folder automatically
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ── Auto-detect folder where this script is saved ─────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'bank_churn.csv')
CHARTS_DIR  = os.path.join(BASE_DIR, 'charts')

# Auto-create charts folder if it doesn't exist
os.makedirs(CHARTS_DIR, exist_ok=True)

# Chart style settings
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 100

print("=" * 55)
print("   🏦 BANK CUSTOMER CHURN — EDA ANALYSIS")
print("=" * 55)
print(f"\n📁 Script folder : {BASE_DIR}")
print(f"📊 Charts saved to: {CHARTS_DIR}")


# ── HELPER FUNCTION ───────────────────────────────────────────
def save_and_show(filename, title=""):
    """Save chart and show it. Close the window to continue."""
    path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    print(f"\n✅ Chart saved: {filename}")
    print("   → Close the chart window to continue...")
    plt.show()   # Opens chart window — close it to go to next step


# ════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("STEP 1 — Loading Data")
print("─" * 55)

if not os.path.exists(DATA_PATH):
    print("\n❌ ERROR: 'bank_churn.csv' not found!")
    print(f"   Please place it in: {BASE_DIR}")
    print("   Download from: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn")
    raise SystemExit

df = pd.read_csv(DATA_PATH)
print(f"\n✅ Data loaded successfully!")
print(f"   Rows    : {df.shape[0]:,}")
print(f"   Columns : {df.shape[1]}")
print(f"\n🔍 Columns:\n   {list(df.columns)}")


# ════════════════════════════════════════════════════════════
# STEP 2 — DATA OVERVIEW
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("STEP 2 — Data Overview")
print("─" * 55)

print("\n📋 Data Types & Non-Null Count:")
print(df.info())

print("\n📊 Basic Statistics (Numeric Columns):")
print(df.describe().round(2))

print("\n🔎 Missing Values:")
print(df.isnull().sum())


# ════════════════════════════════════════════════════════════
# STEP 3 — DATA CLEANING
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("STEP 3 — Data Cleaning")
print("─" * 55)

# Drop identifier columns (not useful for analysis)
cols_to_drop = [c for c in ['RowNumber', 'CustomerId', 'Surname'] if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"\n🗑  Dropped columns: {cols_to_drop}")

# Remove duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"🧹 Duplicates removed: {before - len(df)}")

# Rename target column for clarity
df.rename(columns={'Exited': 'Churned'}, inplace=True)

print(f"✅ Clean dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")


# ════════════════════════════════════════════════════════════
# STEP 4 — CHURN OVERVIEW (Target Variable)
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("STEP 4 — Churn Overview")
print("─" * 55)

counts  = df['Churned'].value_counts()
pct     = df['Churned'].value_counts(normalize=True) * 100

print(f"\n   Stayed  (0): {counts[0]:,}  ({pct[0]:.1f}%)")
print(f"   Churned (1): {counts[1]:,}  ({pct[1]:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('STEP 4 — Churn Distribution', fontsize=14, fontweight='bold')

axes[0].bar(['Stayed', 'Churned'], counts, color=['#2196F3', '#F44336'], width=0.5)
axes[0].set_title('Customer Count')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts):
    axes[0].text(i, v + 50, f"{v:,}", ha='center', fontweight='bold')

axes[1].pie(counts, labels=['Stayed', 'Churned'],
            autopct='%1.1f%%', colors=['#2196F3', '#F44336'],
            startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1].set_title('Churn Rate (%)')

plt.tight_layout()
save_and_show('01_churn_overview.png')


# ════════════════════════════════════════════════════════════
# STEP 5 — UNIVARIATE ANALYSIS (Distributions)
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("STEP 5 — Univariate Analysis (Feature Distributions)")
print("─" * 55)

numeric_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'Tenure', 'NumOfProducts']

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('STEP 5 — Distribution of Key Features', fontsize=14, fontweight='bold')

for i, col in enumerate(numeric_cols):
    ax = axes[i // 3][i % 3]
    sns.histplot(df[col], ax=ax, kde=True, color='steelblue', bins=30)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('')

plt.tight_layout()
save_and_show('02_distributions.png')


# ════════════════════════════════════════════════════════════
# STEP 6 — AGE vs CHURN
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("STEP 6 — Age vs Churn")
print("─" * 55)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('STEP 6 — Age vs Churn', fontsize=14, fontweight='bold')

sns.histplot(data=df, x='Age', hue='Churned', kde=True, bins=30,
             palette={0: '#2196F3', 1: '#F44336'}, ax=axes[0])
axes[0].set_title('Age Distribution by Churn')
axes[0].legend(title='Churned', labels=['No', 'Yes'])

sns.boxplot(data=df, x='Churned', y='Age',
            palette={0: '#2196F3', 1: '#F44336'}, ax=axes[1])
axes[1].set_title('Age Boxplot by Churn')
axes[1].set_xticklabels(['Stayed', 'Churned'])

plt.tight_layout()
save_and_show('03_age_vs_churn.png')


# ════════════════════════════════════════════════════════════
# STEP 7 — GEOGRAPHY vs CHURN
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("STEP 7 — Geography vs Churn")
print("─" * 55)

geo_rate = df.groupby('Geography')['Churned'].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('STEP 7 — Geography vs Churn', fontsize=14, fontweight='bold')

sns.countplot(data=df, x='Geography', hue='Churned',
              palette={0: '#2196F3', 1: '#F44336'}, ax=axes[0])
axes[0].set_title('Count by Country')
axes[0].legend(title='Churned', labels=['No', 'Yes'])

axes[1].bar(geo_rate.index, geo_rate.values,
            color=['#FF9800', '#E91E63', '#9C27B0'], edgecolor='white')
axes[1].set_title('Churn Rate (%) by Country')
axes[1].set_ylabel('Churn Rate (%)')
for i, (idx, v) in enumerate(geo_rate.items()):
    axes[1].text(i, v + 0.3, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
save_and_show('04_geography_vs_churn.png')


# ════════════════════════════════════════════════════════════
# STEP 8 — GENDER vs CHURN
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("STEP 8 — Gender vs Churn")
print("─" * 55)

gender_rate = df.groupby('Gender')['Churned'].mean() * 100
print(f"\n   Female Churn Rate: {gender_rate['Female']:.1f}%")
print(f"   Male Churn Rate  : {gender_rate['Male']:.1f}%")

fig, ax = plt.subplots(figsize=(7, 5))
gender_rate.plot(kind='bar', ax=ax, color=['#EC407A', '#29B6F6'],
                 edgecolor='white', width=0.5)
ax.set_title('STEP 8 — Churn Rate (%) by Gender', fontsize=13, fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
ax.set_xticklabels(['Female', 'Male'], rotation=0)
for i, v in enumerate(gender_rate):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
save_and_show('05_gender_vs_churn.png')


# ════════════════════════════════════════════════════════════
# STEP 9 — ACCOUNT BALANCE vs CHURN
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("STEP 9 — Account Balance vs Churn")
print("─" * 55)

avg_bal_churned = df[df['Churned']==1]['Balance'].mean()
avg_bal_stayed  = df[df['Churned']==0]['Balance'].mean()
print(f"\n   Avg Balance (Churned): {avg_bal_churned:,.0f}")
print(f"   Avg Balance (Stayed) : {avg_bal_stayed:,.0f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('STEP 9 — Balance vs Churn', fontsize=14, fontweight='bold')

sns.histplot(data=df, x='Balance', hue='Churned', kde=True, bins=30,
             palette={0: '#2196F3', 1: '#F44336'}, ax=axes[0])
axes[0].set_title('Balance Distribution by Churn')
axes[0].legend(title='Churned', labels=['No', 'Yes'])

sns.boxplot(data=df, x='Churned', y='Balance',
            palette={0: '#2196F3', 1: '#F44336'}, ax=axes[1])
axes[1].set_title('Balance Boxplot by Churn')
axes[1].set_xticklabels(['Stayed', 'Churned'])

plt.tight_layout()
save_and_show('06_balance_vs_churn.png')


# ════════════════════════════════════════════════════════════
# STEP 10 — PRODUCTS & ACTIVE MEMBER vs CHURN
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("STEP 10 — Products & Active Member vs Churn")
print("─" * 55)

product_rate = df.groupby('NumOfProducts')['Churned'].mean() * 100
active_rate  = df.groupby('IsActiveMember')['Churned'].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('STEP 10 — Products & Active Member vs Churn', fontsize=14, fontweight='bold')

axes[0].bar(product_rate.index.astype(str), product_rate.values,
            color='#FF7043', edgecolor='white')
axes[0].set_title('Churn Rate by No. of Products')
axes[0].set_xlabel('Number of Products')
axes[0].set_ylabel('Churn Rate (%)')
for i, (idx, v) in enumerate(product_rate.items()):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')

active_rate.plot(kind='bar', ax=axes[1],
                 color=['#EF5350', '#66BB6A'], edgecolor='white', width=0.5)
axes[1].set_title('Churn Rate by Active Member Status')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_xticklabels(['Inactive', 'Active'], rotation=0)
for i, v in enumerate(active_rate):
    axes[1].text(i, v + 0.3, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
save_and_show('07_products_active_vs_churn.png')


# ════════════════════════════════════════════════════════════
# STEP 11 — AGE GROUP ANALYSIS
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("STEP 11 — Age Group Analysis")
print("─" * 55)

df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 100],
                         labels=['18-30', '31-40', '41-50', '51-60', '60+'])
age_group_rate = df.groupby('AgeGroup', observed=True)['Churned'].mean() * 100

fig, ax = plt.subplots(figsize=(9, 5))
colors = sns.color_palette("RdYlGn_r", len(age_group_rate))
bars = ax.bar(age_group_rate.index.astype(str), age_group_rate.values,
              color=colors, edgecolor='white')
ax.set_title('STEP 11 — Churn Rate (%) by Age Group', fontsize=13, fontweight='bold')
ax.set_xlabel('Age Group')
ax.set_ylabel('Churn Rate (%)')
for bar, val in zip(bars, age_group_rate.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
save_and_show('08_agegroup_vs_churn.png')


# ════════════════════════════════════════════════════════════
# STEP 12 — CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("STEP 12 — Correlation Heatmap")
print("─" * 55)

numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax, square=True,
            cbar_kws={'shrink': 0.8})
ax.set_title('STEP 12 — Correlation Heatmap', fontsize=14, fontweight='bold')

plt.tight_layout()
save_and_show('09_correlation_heatmap.png')


# ════════════════════════════════════════════════════════════
# STEP 13 — BUSINESS INSIGHTS SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("   💡 KEY BUSINESS INSIGHTS SUMMARY")
print("=" * 55)

total   = len(df)
churned = df['Churned'].sum()

print(f"\n  Overall Churn Rate     : {churned/total*100:.1f}%")
print(f"  Total Customers        : {total:,}")
print(f"  Churned Customers      : {churned:,}")

top_geo      = df.groupby('Geography')['Churned'].mean().idxmax()
top_geo_rate = df.groupby('Geography')['Churned'].mean().max() * 100
print(f"\n  Highest Churn Country  : {top_geo} ({top_geo_rate:.1f}%)")

top_age      = age_group_rate.idxmax()
top_age_rate = age_group_rate.max()
print(f"  Highest Churn Age Group: {top_age} ({top_age_rate:.1f}%)")

f_churn = df[df['Gender']=='Female']['Churned'].mean() * 100
m_churn = df[df['Gender']=='Male']['Churned'].mean() * 100
print(f"\n  Female Churn Rate      : {f_churn:.1f}%")
print(f"  Male Churn Rate        : {m_churn:.1f}%")

act_rate  = df[df['IsActiveMember']==1]['Churned'].mean() * 100
inact_rate = df[df['IsActiveMember']==0]['Churned'].mean() * 100
print(f"\n  Active Member Churn    : {act_rate:.1f}%")
print(f"  Inactive Member Churn  : {inact_rate:.1f}%")

c_bal = df[df['Churned']==1]['Balance'].mean()
s_bal = df[df['Churned']==0]['Balance'].mean()
print(f"\n  Avg Balance (Churned)  : {c_bal:,.0f}")
print(f"  Avg Balance (Stayed)   : {s_bal:,.0f}")

print("\n" + "=" * 55)
print(f"  ✅ All charts saved in: {CHARTS_DIR}")
print("=" * 55)
