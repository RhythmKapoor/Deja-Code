
import pandas as pd

DATA_PATH = "data/questions (1).csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df[["QCode", "Title", "statement", "Tags"]].isna().sum())

print("\nFirst five problems:")
print(df[["QCode", "Title", "level"]].head())

# Keep only rows with usable problem statements
df = df.dropna(subset=["statement", "Title", "QCode"])

# Remove duplicate problem statements
df = df.drop_duplicates(subset=["statement"])

print("\nProblems after cleaning:", len(df))
