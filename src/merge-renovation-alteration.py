import csv

from utils import TESTING_DATA_CSV, TRAINING_DATA_CSV

updated_train_data = []
updated_test_data = []

with (
    open(TRAINING_DATA_CSV, mode="r", encoding="utf-8", newline="") as train_file,
    open(TESTING_DATA_CSV, mode="r", encoding="utf-8", newline="") as test_file,
):
    train_data = csv.reader(train_file)
    test_data = csv.reader(test_file)

    for row in train_data:
        content, page, type, file = row

        if type == "type":
            updated_train_data += [row]
            continue

        updated_type = int(type)
        if updated_type > 5:
            updated_type -= 1

        updated_train_data += [(content, page, updated_type, file)]

    for row in test_data:
        content, page, type, file = row

        if type == "type":
            updated_test_data += [row]
            continue

        updated_type = int(type)
        if updated_type > 5:
            updated_type -= 1

        updated_test_data += [(content, page, updated_type, file)]


with open(TRAINING_DATA_CSV, mode="w", encoding="utf-8", newline="") as train_file:
    writer = csv.writer(train_file)

    for row in updated_train_data:
        writer.writerow(row)

with open(TESTING_DATA_CSV, mode="w", encoding="utf-8", newline="") as test_file:
    writer = csv.writer(test_file)

    for row in updated_test_data:
        writer.writerow(row)
