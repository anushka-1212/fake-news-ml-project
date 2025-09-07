# combine_csv.py
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent
data_dir = root / "data"

# 1) Read the two Kaggle files
fake = pd.read_csv(data_dir / "Fake.csv")
true = pd.read_csv(data_dir / "True.csv")

# 2) Build one clean "text" column (title + text). If a column is missing, use empty string.
def make_text(df):
    title = df["title"] if "title" in df.columns else ""
    text = df["text"] if "text" in df.columns else ""
    return (title.fillna("") + ". " + text.fillna("")).str.strip()

fake_df = pd.DataFrame({
    "text": make_text(fake),
    "label": 0  # Fake = 0
})
true_df = pd.DataFrame({
    "text": make_text(true),
    "label": 1  # Real = 1
})

# 3) Combine + shuffle (for good measure)
combined = pd.concat([fake_df, true_df], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# 4) Save as one CSV used by the training script
out_path = data_dir / "fake_news.csv"
combined.to_csv(out_path, index=False)

print(f"✅ Saved {len(combined)} rows to {out_path}")
print("Columns:", list(combined.columns))
print(combined.head(3))
