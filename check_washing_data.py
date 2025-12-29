import pandas as pd

df = pd.read_csv('created_data/UK_DALE/washingmachine_training_.csv')

print('='*60)
print('WASHING MACHINE TRAINING DATA ANALYSIS')
print('='*60)
print(f'\nColumns: {df.columns.tolist()}')
print(f'Shape: {df.shape}')

print('\n' + '='*60)
print('WASHING MACHINE POWER STATISTICS')
print('='*60)
print(f'Min: {df["washingmachine"].min():.6f}')
print(f'Max: {df["washingmachine"].max():.6f}')
print(f'Mean: {df["washingmachine"].mean():.6f}')
print(f'Std: {df["washingmachine"].std():.6f}')

print('\n' + '='*60)
print('FIRST 20 VALUES')
print('='*60)
print(df["washingmachine"].head(20).values)

print('\n' + '='*60)
print('VALUE DISTRIBUTION')
print('='*60)
print(f'Values < 0: {(df["washingmachine"] < 0).sum()}')
print(f'Values = 0: {(df["washingmachine"] == 0).sum()}')
print(f'Values > 0 and < 1: {((df["washingmachine"] > 0) & (df["washingmachine"] < 1)).sum()}')
print(f'Values >= 1: {(df["washingmachine"] >= 1).sum()}')
print(f'Values >= 10: {(df["washingmachine"] >= 10).sum()}')

# Check if this is Z-score or something else
print('\n' + '='*60)
print('NORMALIZATION TYPE CHECK')
print('='*60)
if df["washingmachine"].min() >= 0 and df["washingmachine"].max() <= 1:
    print('Looks like MinMax [0,1] normalization')
elif df["washingmachine"].min() < 0 and df["washingmachine"].max() > 0:
    print('Looks like Z-score normalization')
else:
    print('Unknown normalization type')
