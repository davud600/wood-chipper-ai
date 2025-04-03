import csv

from src.utils import (
    TRAINING_DATA_CSV,
    TESTING_DATA_CSV,
    CORPUS_FILE,
)


content = ""

with (
    open(TRAINING_DATA_CSV, mode="r", encoding="utf-8") as train_file,
    open(TESTING_DATA_CSV, mode="r", encoding="utf-8") as test_file,
):
    train_reader = csv.reader(train_file)
    test_reader = csv.reader(test_file)

    for row in train_reader:
        if row[0] == "content":  # skip headers.
            continue

        content += str(row[0])
    for row in test_reader:
        if row[0] == "content":  # skip headers.
            continue

        content += str(row[0])

with open(CORPUS_FILE, mode="w", encoding="utf-8") as file:
    file.write(content)
