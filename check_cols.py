import pandas as pd
df = pd.read_csv('./Data/datasets/fridge_multivariate.csv')
print(f"Total columns: {len(df.columns)}")
for i, col in enumerate(df.columns):
    print(f"Col {i}: '{col}'")
