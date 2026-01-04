import pandas as pd

df = pd.read_csv('FinalTrainingData3.csv')

print("=" * 60)
print("FLOOD COUNT ANALYSIS")
print("=" * 60)
print("\nBasic Statistics:")
print(df['flood_count'].describe())

print("\n" + "=" * 60)
print("Value Counts (flood_count):")
print(df['flood_count'].value_counts().sort_index())

print("\n" + "=" * 60)
print("Distribution by Percentage:")
vc = df['flood_count'].value_counts(normalize=True).sort_index() * 100
for val, pct in vc.items():
    print(f"  {val}: {pct:.2f}%")

print("\n" + "=" * 60)
print("2025 Data Analysis:")
df_2025 = df[df['year'] == 2025]
print(f"Total 2025 records: {len(df_2025)}")
print("\n2025 Flood count stats:")
print(df_2025['flood_count'].describe())
print("\n2025 Value counts:")
print(df_2025['flood_count'].value_counts().sort_index())

print("\n" + "=" * 60)
print("Training Data (2015-2022) Analysis:")
df_train = df[(df['year'] >= 2015) & (df['year'] <= 2022)]
print(f"Total training records: {len(df_train)}")
print("\nTraining Flood count stats:")
print(df_train['flood_count'].describe())
print("\nTraining Value counts:")
print(df_train['flood_count'].value_counts().sort_index())

print("\n" + "=" * 60)
print("HIGH RISK RECORDS (flood_count >= 3):")
high_risk = df[df['flood_count'] >= 3]
print(f"Count: {len(high_risk)}")
if len(high_risk) > 0:
    print("\nSample high-risk records:")
    print(high_risk[['Ward_No', 'WardName', 'year', 'flood_count', 'max_rainfall_3day_mm']].head(20))
