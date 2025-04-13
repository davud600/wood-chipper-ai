import pandas as pd
from config.settings import TRAINING_DATA_CSV, TESTING_DATA_CSV, DOCUMENT_TYPES


train_total_pages = 0
train_type_counters = {
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
    12: 0,
    13: 0,
    14: 0,
    15: 0,
}
test_total_pages = 0
test_type_counters = {
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
    12: 0,
    13: 0,
    14: 0,
    15: 0,
}


def check_data_leakage(train_csv_path, test_csv_path):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_files = set(train_df["file"].unique())
    test_files = set(test_df["file"].unique())

    overlap = train_files.intersection(test_files)

    print(f"🔍 Total unique train files: {len(train_files)}")
    print(f"🔍 Total unique test files:  {len(test_files)}")
    print(f"🚨 Overlapping files: {len(overlap)}")

    if overlap:
        print("⚠️  Example overlapping files:")
        for file in list(overlap)[:10]:
            print(f"  - {file}")
    else:
        print("✅ No data leakage detected. Train and test are clean.")


if __name__ == "__main__":
    check_data_leakage(TRAINING_DATA_CSV, TESTING_DATA_CSV)

    train_data = pd.read_csv(TRAINING_DATA_CSV)
    train_sum = 0
    test_data = pd.read_csv(TESTING_DATA_CSV)
    test_sum = 0

    for doc_type in train_data["type"].tolist():
        train_sum += 1
        train_type_counters[int(doc_type)] += 1
    for doc_type in test_data["type"].tolist():
        test_sum += 1
        test_type_counters[int(doc_type)] += 1

    for t, doc_type in enumerate(list(DOCUMENT_TYPES.keys())):
        print(f"{doc_type}: {train_type_counters[t]}")

    print(f"train total: {train_sum}")

    for t, doc_type in enumerate(list(DOCUMENT_TYPES.keys())):
        print(f"{doc_type}: {test_type_counters[t]}")

    print(f"test total: {test_sum}")
