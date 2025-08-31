import pandas as pd

# 1. Load your full dataset
df = pd.read_csv("lending_club_loan_two.csv")

# 2. Take a random sample of 5000 rows
sample_df = df.sample(n=5000, random_state=42)

# 3. Save to new folder 'data/' as sample_lending_club.csv
sample_df.to_csv("data/sample_lending_club.csv", index=False)

print("✅ Sample dataset created at: data/sample_lending_club.csv")
