import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# styling plots 
plt.style.use('seaborn-v0_8')

# Load Data
train = pd.read_csv('train.csv')
valid   = pd.read_csv('valid.csv')
test  = pd.read_csv('test.csv')

print("Train size:", len(train))
print("Validation size:", len(valid))
print("Test size:", len(test))

# print columns we have
print("\nColumns:", train.columns.tolist())
train.head(3)

print("\n")

#checking labels
print("------ HAZARD CATEGORIES ------")
print(train['hazard-category'].value_counts())

print("\n")

print("------ PRODUCT CATEGORIES ------")
print(train['product-category'].value_counts())

print("Sample titles:\n")
for i in range(5):
    print(f"Title: {train['title'].iloc[i]}")
    print(f"  hazard-category : {train['hazard-category'].iloc[i]}")
    print(f"  product-category: {train['product-category'].iloc[i]}")
    print()


#PLOTS needs fixing 

"""
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# Hazard category distribution
train['hazard-category'].value_counts().plot(
    kind='bar', ax=axes[0], color='steelblue'
)
axes[0].set_title('Hazard Category Distribution (Train)')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=45)

# Product category distribution
train['product-category'].value_counts().plot(
    kind='bar', ax=axes[1], color='coral'
)
axes[1].set_title('Product Category Distribution (Train)')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
"""

#show the inbalance visualy
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Hazard
hazard_counts = train['hazard-category'].value_counts()
colors_h = ['#d32f2f' if c < 100 else '#ff9800' if c < 500 else '#388e3c' 
            for c in hazard_counts.values]
hazard_counts.plot(kind='bar', ax=axes[0], color=colors_h)
axes[0].set_title('Hazard Category Distribution', fontsize=13)
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylabel('Number of examples')

# Product
product_counts = train['product-category'].value_counts()
colors_p = ['#d32f2f' if c < 20 else '#ff9800' if c < 100 else '#388e3c' 
            for c in product_counts.values]
product_counts.plot(kind='bar', ax=axes[1], color=colors_p)
axes[1].set_title('Product Category Distribution', fontsize=13)
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylabel('Number of examples')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Print imbalance ratio
print(f"Hazard imbalance ratio: {hazard_counts.max()} vs {hazard_counts.min()} "
      f"({hazard_counts.max()//hazard_counts.min()}x difference)")
print(f"Product imbalance ratio: {product_counts.max()} vs {product_counts.min()} "
      f"({product_counts.max()//product_counts.min()}x difference)")

# How long are the titles?
train['title_length'] = train['title'].str.split().str.len()

print("Title length stats:")
print(train['title_length'].describe())

# Are some hazard categories associated with longer titles?
print("\nAverage title length per hazard category:")
print(train.groupby('hazard-category')['title_length'].mean().sort_values(ascending=False))

# Compare title vs full text length
train['text_length'] = train['text'].str.split().str.len()

print("TEXT column length stats:")
print(train['text_length'].describe())

print("\nSample of full 'text' for first 2 rows:")
for i in range(2):
    print(f"\n--- Example {i+1} ---")
    print(f"TITLE: {train['title'].iloc[i]}")
    print(f"TEXT (first 200 chars): {str(train['text'].iloc[i])[:200]}")
    print(f"hazard-category: {train['hazard-category'].iloc[i]}")


# See more of the full text
print(train['text'].iloc[0])

recent = train[train['year'] > 2015].iloc[0]
print(f"TITLE: {recent['title']}")
print(f"\nTEXT: {recent['text'][:500]}")
print(f"\nhazard-category: {recent['hazard-category']}")
print(f"product-category: {recent['product-category']}")