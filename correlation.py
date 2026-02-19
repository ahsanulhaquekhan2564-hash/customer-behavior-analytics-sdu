from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the cleaned data
df = pd.read_csv("exam_cleaned.csv")

# 2. Data Summary (Faculty Pattern)
summary_df = pd.DataFrame({
    'mean' : df.select_dtypes(include='number').mean(),
    'std' : df.select_dtypes(include='number').std(),
    'min' : df.select_dtypes(include='number').min(),
    'max' : df.select_dtypes(include='number').max(),
    'median' : df.select_dtypes(include='number').median(),
    'missing' : df.isnull().sum()
})

print(" Data Summary Table (All Numeric Columns)")
print(summary_df)

# 3. Correlation Matrix for ALL numeric columns
corr_df = df.select_dtypes(include='number').corr()

# 4. Visualization (Replicating the Reference Style)
# Setting the figure size and white background
plt.figure(figsize=(22, 18), facecolor='white')

# Plotting the heatmap with specific cube borders and font styles
sns.heatmap(
    corr_df, 
    annot=True,       
    fmt=".2f",
    cmap="coolwarm",           # Red-blue diverging palette
    annot_kws={"size": 5, "color": "black", "weight": "normal"}, # Smaller numbers
    linewidths=1.5,            # Thicker lines to make the "cubes" very distinct
    linecolor='#E0E0E0',       # Light gray borders between cells (the grid)
    square=True,               # Forces every cell to be a perfect cube
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
)

# 5. Header / Title - Bold and clearly visible at the top center
plt.title("Cleaned Dataset correlation Heatmap)", 
          fontsize=26, 
          fontweight='bold', 
          pad=30) 

# 6. Formatting Labels (Rotated for readability)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

# tight_layout ensures everything is centered and labels are not cut off
plt.tight_layout()

# Save the final version as a high-quality PNG
plt.savefig("exam_final_grid_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Final grid heatmap saved as: exam_final_grid_heatmap.png")