import pandas as pd

# Load your full dataset
df = pd.read_csv("lending_club_loan_two.csv")

# Take a random sample of 5000 rows (you can change number)
sample_df = df.sample(n=5000, random_state=42)

# Save to new file
sample_df.to_csv("data/sample_lending_club.csv", index=False)

print("✅ Sample dataset created: data/sample_lending_club.csv")
