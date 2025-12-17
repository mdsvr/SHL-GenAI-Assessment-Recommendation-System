import pandas as pd

FILE_PATH = "Gen_AI Dataset.xlsx"

print("Loading Excel file...\n")

# Load Excel
df = pd.read_excel(FILE_PATH)

print("✅ File loaded successfully\n")

# Basic info
print("===== BASIC INFO =====")
print(f"Number of rows   : {len(df)}")
print(f"Number of columns: {len(df.columns)}\n")

# Column names
print("===== COLUMN NAMES =====")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")
print()

# Data types
print("===== COLUMN DATA TYPES =====")
print(df.dtypes)
print()

# First 3 rows (raw view)
print("===== FIRST 3 ROWS =====")
print(df.head(3))
print()

# Inspect unique values length for second column
if len(df.columns) >= 2:
    print("===== SAMPLE OF RELEVANT URL FIELD =====")
    sample = df.iloc[0, 1]
    print(sample)
    print("\nType:", type(sample))
