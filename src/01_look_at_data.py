import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" 

# finding every .csv file under DATA_DIR and storing the list in a variable called files
files=list(DATA_DIR.rglob("*.csv"))
print("File Names :-")
for file in files:
    print(file.stem)

# number of files found
print(len(files)," files were found");

# loading the first file into a pandas DataFrame called df
df=pd.read_csv(files[0])
print("First 5 rows of ",files[0].name)
print(df.head())

# printing df.shape
print("File Shape= ",df.shape)
print("File Dimension",df.ndim)

# printing the first 12 column names
print("First 12 column names- ",df.columns[:12])

print(files[-1].name)
df=pd.read_csv(files[-1])
print(df)

print(files[-2].name)
df=pd.read_csv(files[-2])
print(df)

# data summary
print(files[-3].name)
df=pd.read_csv(files[-3])
print(df)
df.columns = df.columns.str.strip()
print(df.columns)

# counting benign rows
benign=df[df["File Name"].str.contains("benign")]
benign_count=benign["Data Count"].sum()
print("Benign rows- ",benign_count)

# counting gafgyt rows
gafgyt=df[df["File Name"].str.contains("gafgyt")]
gafgyt_count=gafgyt["Data Count"].sum()
print("Gafgyt rows- ",gafgyt_count)

# counting mirai rows
mirai_count=df[df["File Name"].str.contains("mirai")]["Data Count"].sum()
print("Mirai rows- ",mirai_count)