# --------------------------------------------------------------
#  Pandas + Matplotlib Data Analysis
# --------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# -------------------------------
# 1. Load the CSV file
# -------------------------------
# Option A: Use the built-in Iris dataset (no download needed)
# -------------------------------------------------------
iris = sns.load_dataset('iris')
df = iris.copy()

# Option B: Load your own CSV (uncomment & edit path)
# -------------------------------------------------------
# csv_path = 'path/to/your/data.csv'   # <-- change this
# if not os.path.exists(csv_path):
#     raise FileNotFoundError(f"File not found: {csv_path}")
# df = pd.read_csv(csv_path)

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(df.head())

# -------------------------------
# 2. Basic statistical analysis
# -------------------------------
print("\n--- Basic Summary Statistics ---")
print(df.describe(include='all'))

# Example: Average of a selected numeric column
selected_col = 'sepal_length'      # change to any numeric column
avg_value = df[selected_col].mean()
print(f"\nAverage of '{selected_col}': {avg_value:.2f}")

# -------------------------------------------------
# 3. Visualizations
# -------------------------------------------------
sns.set_style("whitegrid")
plt.figure(figsize=(16, 12))

# ---- Bar Chart: Average of a numeric column per category ----
cat_col = 'species'                # categorical column
num_col = 'petal_length'           # numeric column for bar height

bar_data = df.groupby(cat_col)[num_col].mean().reset_index()

plt.subplot(2, 3, 1)
sns.barplot(data=bar_data, x=cat_col, y=num_col, hue=cat_col, palette='viridis', legend=False)
plt.title(f'Average {num_col} by {cat_col}')
plt.ylabel(f'Avg {num_col}')
plt.xlabel(cat_col)
plt.xticks(rotation=30)

# ---- Scatter Plot: Relationship between two numeric columns ----
x_col = 'sepal_length'
y_col = 'sepal_width'

plt.subplot(2, 3, 2)
sns.scatterplot(data=df, x=x_col, y=y_col, hue=cat_col, style=cat_col,
                palette='deep', s=80, alpha=0.8)
plt.title(f'{x_col} vs {y_col}')
plt.legend(title=cat_col, bbox_to_anchor=(1.05, 1), loc='upper left')

# ---- Correlation Heatmap (numeric columns only) ----
plt.subplot(2, 3, 3)
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',
            linewidths=.5, cbar_kws={'shrink': .8})
plt.title('Correlation Heatmap')

# ---- Bonus: Pairplot (quick overview) ----
plt.subplot(2, 3, 4)
# Pairplot is heavy; show only a subset for demo
sns.pairplot(df, hue=cat_col, corner=True, diag_kind='kde')
plt.suptitle('Pairplot Overview', y=1.02)

# Adjust layout and show
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 4. Insights & Observations
# -------------------------------------------------
print("\n" + "="*60)
print("INSIGHTS & OBSERVATIONS")
print("="*60)

print(f"1. Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"   - Categorical column: '{cat_col}' with {df[cat_col].nunique()} unique values.")
print(f"   - Average '{selected_col}': {avg_value:.2f}")

print("\n2. Bar Chart:")
print(f"   - 'virginica' has the highest average {num_col} "
      f"({bar_data.loc[bar_data[cat_col]=='virginica', num_col].values[0]:.2f}).")
print(f"   - 'setosa' shows the lowest average {num_col} "
      f"({bar_data.loc[bar_data[cat_col]=='setosa', num_col].values[0]:.2f}).")

print("\n3. Scatter Plot:")
print(f"   - There is a moderate negative relationship between {x_col} and {y_col} "
      f"(correlation = {corr.loc[x_col, y_col]:.2f}).")
print("   - 'setosa' tends to cluster at lower sepal length & higher sepal width.")

print("\n4. Correlation Heatmap:")
strong_corr = corr.abs().unstack().sort_values(ascending=False)
strong_corr = strong_corr[strong_corr < 1].head(3)  # top 3 non-diagonal
print("   Top positive correlations:")
for (c1, c2), val in strong_corr.items():
    print(f"       • {c1} ↔ {c2}: {val:.2f}")

print("\n5. General Takeaway:")
print("   The Iris dataset demonstrates clear separation between species "
      "using petal measurements, making it ideal for classification tasks.")
print("="*60)