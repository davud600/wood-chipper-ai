import random
import csv


from config.settings import DOCUMENT_TYPES, pages_to_append
from type_defs import Dataset


def get_dataset(path: str, mini_batch_size: int) -> tuple[Dataset, int, int]:
    data: list[tuple[str, int, int, str]] = []
    dataset: Dataset = []
    contents: list[str] = []
    pages: list[int] = []
    types: list[int] = []
    N0 = 0
    N1 = 0

    with open(file=path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)

        for r, row in enumerate(reader):
            if r == 0:  # skip headers.
                continue

            data += [(str(row[0]), int(row[1]), int(row[2]), str(row[3]))]

    type_counters = [
        {"first_page": 0, "not_first_page": 0}
        for _ in range(len(DOCUMENT_TYPES.keys()))
    ]

    for r, row in enumerate(data):
        content = row[0]
        page = row[1]
        type = row[2]
        file = row[3]

        non_first_pages_prob = 1  # bigger -> more non-first pages.
        if page != 1 and random.random() > non_first_pages_prob:
            continue

        content = f"<curr_page>{content}</curr_page>"
        for next in range(1, pages_to_append + 1):
            if r - next < 0 or data[r - next][3] != file:
                break

            content += f"<next_page_{next}>{data[r - next][0]}</next_page_{next}>"

        pages += [page]
        types += [type]
        contents += [content]
        if page == 1:
            type_counters[type]["first_page"] += 1
        else:
            type_counters[type]["not_first_page"] += 1

    zipped_data = list(zip(contents, pages))
    random.shuffle(zipped_data)
    shuffled_contents, shuffled_pages = zip(*zipped_data)
    mini_batch_features: list[str] = []
    mini_batch_labels: list[int] = []
    counter = 0
    for features, labels in zip(shuffled_contents, shuffled_pages):
        if counter >= mini_batch_size:
            dataset.append(
                {
                    "features": mini_batch_features,
                    "labels": mini_batch_labels,
                }
            )

            mini_batch_features = []
            mini_batch_labels = []
            counter = 0

        mini_batch_features.append(features)
        mini_batch_labels.append(labels)
        counter += 1

    if mini_batch_features:
        dataset.append(
            {
                "features": mini_batch_features,
                "labels": mini_batch_labels,
            }
        )

    N0 = 0
    N1 = 0
    for t, document_type in enumerate(list(DOCUMENT_TYPES.keys())):
        N0 += type_counters[t]["not_first_page"]
        N1 += type_counters[t]["first_page"]

        print(
            f"{document_type}: {type_counters[t]['first_page'] + type_counters[t]['not_first_page']} ({type_counters[t]['first_page']}, {type_counters[t]['not_first_page']})"
        )

    return dataset, N0, N1
