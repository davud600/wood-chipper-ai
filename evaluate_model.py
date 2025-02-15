import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from train_model import MultiTaskBertModel

def predict(text, model, tokenizer, max_length=512, device="cpu"):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits_first_page = outputs['logits_first_page']
    logits_type = outputs['logits_type']
    probabilities_first_page = torch.softmax(logits_first_page, dim=1)
    probabilities_type = torch.softmax(logits_type, dim=1)

    return probabilities_first_page, probabilities_type

def evaluate_model(predictions_first_page, predictions_type, true_labels_first_page, true_labels_type, texts, file_names, page_numbers):
    true_labels_first_page = np.array(true_labels_first_page).flatten()
    true_labels_type = np.array(true_labels_type).flatten()

    print("Classification Report for First Page:")
    print(classification_report(true_labels_first_page, predictions_first_page, labels=[0, 1]))
    print("Confusion Matrix for First Page:")
    print(confusion_matrix(true_labels_first_page, predictions_first_page, labels=[0, 1]))

    print("Classification Report for Type:")
    print(classification_report(true_labels_type, predictions_type))
    print("Confusion Matrix for Type:")
    print(confusion_matrix(true_labels_type, predictions_type))

    misclassified_rows = []
    for i, (pred_fp, label_fp, pred_type, label_type) in enumerate(zip(predictions_first_page, true_labels_first_page, predictions_type, true_labels_type)):
        if pred_fp != label_fp or pred_type != label_type:
            misclassified_rows.append({
                "file_name": file_names[i],
                "true_first_page": label_fp,
                "predicted_first_page": pred_fp,
                "true_type": label_type,
                "predicted_type": pred_type,
                "page_number": page_numbers[i],
                "text": texts[i]
            })

    with open('misclassified_rows.txt', 'w', encoding='utf-8') as f:
        for row in misclassified_rows:
            f.write(f"File: {row['file_name']}, Page: {row['page_number']}, True First Page: {row['true_first_page']}, Pred First Page: {row['predicted_first_page']}, True Type: {row['true_type']}, Pred Type: {row['predicted_type']}, Text: {row['text']}\n")

    print(f"Logged {len(misclassified_rows)} misclassified rows to 'misclassified_rows.txt'.")

if __name__ == "__main__":
    model_output_dir = "multi_task_bert_model"
    testing_data_csv = "testing_data.csv"
    if not os.path.isdir(model_output_dir):
        raise FileNotFoundError(f"The directory {model_output_dir} does not exist.")
    if not os.path.isfile(testing_data_csv):
        raise FileNotFoundError(f"The file {testing_data_csv} does not exist.")

    tokenizer = BertTokenizer.from_pretrained(model_output_dir)
    num_types = pd.read_csv(testing_data_csv)['type'].nunique()  # Dynamically determine the number of types
    base_model = BertModel.from_pretrained("bert-base-uncased")
    model = MultiTaskBertModel(base_model, num_types=num_types)
    state_dict = torch.load(os.path.join(model_output_dir, "pytorch_model.bin"))
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = pd.read_csv(testing_data_csv)
    if 'text' not in data.columns or 'is_first_page' not in data.columns or 'type' not in data.columns:
        raise ValueError("CSV must contain 'text', 'is_first_page', and 'type' columns.")

    data['text'] = data['text'].astype(str).fillna("")
    texts = data['text'].tolist()
    labels_first_page = data['is_first_page'].tolist()
    labels_type = data['type'].tolist()
    file_names = data['file_name'].tolist()
    page_numbers = data['page_number'].tolist()

    predictions_first_page = []
    predictions_type = []
    valid_labels_first_page = []
    valid_labels_type = []
    valid_texts = []
    valid_file_names = []
    valid_page_numbers = []

    for i, text in enumerate(texts):
        if not isinstance(text, str) or text.strip() == "":
            print(f"Skipping invalid text: {text}")
            continue

        probs_fp, probs_type = predict(text, model, tokenizer, device=device)
        predictions_first_page.append(np.argmax(probs_fp.cpu().numpy()))
        predictions_type.append(np.argmax(probs_type.cpu().numpy()))
        valid_labels_first_page.append(labels_first_page[i])
        valid_labels_type.append(labels_type[i])
        valid_texts.append(text)
        valid_file_names.append(file_names[i])
        valid_page_numbers.append(page_numbers[i])

    predictions_first_page = np.array(predictions_first_page)
    predictions_type = np.array(predictions_type)
    evaluate_model(
        predictions_first_page,
        predictions_type,
        valid_labels_first_page,
        valid_labels_type,
        valid_texts,
        valid_file_names,
        valid_page_numbers
    )
