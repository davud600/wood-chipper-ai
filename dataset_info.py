import pandas as pd

TYPES = {
    "unknown": 0,
    "original-lease": 1,
    "lease-renewal": 2,
    "closing-document": 3,
    "sublease": 4,
    "alteration-document": 5,
    "renovation-document": 6,
    "proprietary-lease": 7,
    "purchase-application": 8,
    "refinance-document": 9,
    "tenant-correspondence": 10,
    "transfer-document": 11,
}

csv_file = "training_data.csv"
data = pd.read_csv(csv_file)

total_pages = 0
type_counters = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
}
sum = 0
for type in data["type"].tolist():
    sum += 1
    type_counters[int(type)] += 1

for t, type in enumerate(list(TYPES.keys())):
    print(f"{type}: {type_counters[t]}")
print(f"total: {sum}")
