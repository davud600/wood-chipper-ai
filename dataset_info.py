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
for type in data["type"].tolist():
    type_counters[int(type)] += 1

text_lens = []
for text in data["text"].tolist():
    text_lens += [len(str(text))]
# print(f"text lengths ({len(text_lens)}): {text_lens}")
print(f"max text length: {max(text_lens)}")

for type in list(type_counters.keys()):
    total_pages += type_counters[type]
    print(f"{list(TYPES.keys())[int(type)]}: {type_counters[type]}")
print(f"total pages: {total_pages}")
