import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR=Path(__file__).resolve().parent.parent /"data"
LABELS={"benign":0,"mirai":1,"gafgyt":2}

# extract class name
def parse_name(path):
    parts=path.stem.split(".")
    return parts[0],parts[1]

# Storing files except helper files (data_summary, device_info, features)
traffic_files=sorted(
    path for path in DATA_DIR.rglob("*.csv")
    if path.stem.split(".")[0].isdigit()
)
"""
traffic_files = []

for path in DATA_DIR.rglob("*.csv"):
    if path.stem.split(".")[0].isdigit():
        traffic_files.append(path)

traffic_files = sorted(traffic_files)
"""
# print the number of files
print(len(traffic_files),"traffic files found")

rows_per_class = {0: 0, 1: 0, 2: 0}
total_missing = 0
total_inf = 0

# accessing indivisual files
for path in traffic_files:
    device, class_name = parse_name(path)
    df = pd.read_csv(path)
    # counting rows and adding them to corresponding class
    rows_per_class[LABELS[class_name]]+=df.shape[0]
    # Count missing values
    total_missing += df.isna().sum().sum()
    # Count infinite values
    total_inf += np.isinf(df.to_numpy()).sum()

    print("\n"+path.stem,"has",df.shape[0],"rows")
    print(path.stem,"has",total_missing,"missing values")
    print(path.stem,"has",total_inf,"infinite values")

# print final result
print("\n========== DATASET SUMMARY ==========")

print("Benign rows:", rows_per_class[0])
print("Mirai rows:", rows_per_class[1])
print("Gafgyt rows:", rows_per_class[2])

total_rows = sum(rows_per_class.values())

print("\nTotal rows:", total_rows)
print("Total missing values:", total_missing)
print("Total infinite values:", total_inf)

print("=====================================")

print("---------Percentages---------")
print("Benign:",(rows_per_class[0]/total_rows)*100) 
print("Attack:",((rows_per_class[1]+rows_per_class[2])/total_rows)*100)    
print("------------------------------")