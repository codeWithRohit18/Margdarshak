import pandas as pd
from db import get_db_connection

# 1. Data Load
df=pd.read_csv("Raw-Dataset/Future of Jobs AI Datset Dirty.csv")
print(df.head(3))

# Start For EDA(Exploratory Data Analysis)

# 2. Data Types
print("\nData Types:")
print(df.dtypes)

# 3. Column
print("\nColumn Name:")
print(df.columns)

# 4. Info 
print("\nInfo Details")
print(df.info())

# 5. Unique Values
print("\nUnique Value")
print(df.nunique())

# Start For Data Preprocessing 

# Data Clean

# 1. Duplicate
print("\nDuplicates")
print(df.duplicated().sum())
print(df.drop_duplicates(inplace=True))
# Data Clean
print(df.duplicated().sum())

# 2. Handel Missing Value
print("\nMissing Value:")
print(df.isnull().sum())
print(df.dropna(inplace=True))
# Data Clean
print(df.isnull().sum())

# 3. Sort job_title Column
df['job_title'] = df['job_title'].str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)
print(df.head(5))

# 4. Handel Upper Case Data
df['education_level'] = df['education_level'].str.capitalize()

# 5. Clean Exprience Level Column
df['experience_level'] = df['experience_level'].str.capitalize()
print(df.head(5))
df['experience_level'] = df['experience_level'].str.replace(r'[^a-zA-Z0-9]', '', regex=True)

# Remove spaces
df['salary'] = df['salary'].astype(str).str.strip()
# Remove invalid characters (keep only digits and dot)
df['salary'] = df['salary'].str.replace(r'[^0-9.]', '', regex=True)
# Fix multiple dots (keep only first one)
df['salary'] = df['salary'].str.replace(r'\.(?=.*\.)', '', regex=True)

# MySql Data Insert Query

conn=get_db_connection()
cur=conn.cursor()

insert_query = """
INSERT INTO jobs_data(
    job_title, country, experience_level, education_level, year,
    salary, ai_risk_score, primary_skill, skill_demand_score,
    job_openings, job_survival_class, salary_bucket, ai_risk_category
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

data = [tuple(row) for row in df.to_numpy()]

cur.executemany(insert_query, data)
conn.commit()

cur.close()
conn.close()

print("✅ Data inserted successfully!")

# Clean Data File 
output="output/handelEDA.csv"
df.to_csv(output,index=True)