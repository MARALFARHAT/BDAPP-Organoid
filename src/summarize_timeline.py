import pandas as pd
from pathlib import Path

path = Path(r"F:\BDAPP-Organoid\outputs\fs437_6hour_event_counts.csv")
df = pd.read_csv(path)

df["window_start"] = pd.to_datetime(df["window_start"])
df = df.sort_values("window_start").reset_index(drop=True)

print("ROWS:", len(df))
print("START:", df["window_start"].min())
print("END:", df["window_start"].max())
print()

print("EVENT COUNT SUMMARY")
print(df["detected_events"].describe())
print()

print("FIRST 10 ROWS")
print(df.head(10).to_string(index=False))
print()

print("LAST 10 ROWS")
print(df.tail(10).to_string(index=False))
print()

n = len(df)
one = n // 3
two = 2 * n // 3

begin_mean = df.iloc[:one]["detected_events"].mean()
middle_mean = df.iloc[one:two]["detected_events"].mean()
end_mean = df.iloc[two:]["detected_events"].mean()

print("MEAN EVENTS PER 6-HOUR WINDOW")
print("Beginning:", round(begin_mean, 2))
print("Middle   :", round(middle_mean, 2))
print("End      :", round(end_mean, 2))
