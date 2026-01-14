import pandas as pd
import numpy as np

filepath = './Data/datasets/fridge_multivariate.csv'
df = pd.read_csv(filepath, header=0)
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
print(f"First row: {df.values[0]}")
