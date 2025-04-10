import pandas as pd

from config.settings import TRAINING_DATA_CSV, TESTING_DATA_CSV


training_data = pd.read_csv(TRAINING_DATA_CSV)
training_files = []
for file in training_data["file"].tolist():
    training_files += [file]

testing_data = pd.read_csv(TESTING_DATA_CSV)
testing_files = []
for file in testing_data["file"].tolist():
    testing_files += [file]

print(training_files[:5])
print(testing_files[:5])

for training_file in training_files:
    if training_file in testing_files:
        print("oops", training_file)
