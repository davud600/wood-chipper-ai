import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

def predict(text, model, tokenizer, max_length=512, device="cpu"):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    return probabilities

def evaluate_model(predictions, true_labels, texts, file_names, page_numbers):
    preds = np.argmax(predictions, axis=1)
    print("Classification Report:")
    print(classification_report(true_labels, preds, labels=[0, 1]))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, preds, labels=[0, 1]))

    misclassified_rows = []
    predictions = predictions.argmax(axis=-1).tolist()
    for i, (pred, label) in enumerate(zip(predictions, true_labels)):
        if pred != label:
            misclassified_rows.append({
                "text": texts[i],
                "true_label": label,
                "predicted_label": pred,
                "file_name": file_names[i],
                "page_number": page_numbers[i]
            })
            
    with open('misclassified_rows.txt', 'w', encoding='utf-8') as f:
        for row in misclassified_rows:
            f.write(f"File: {row['file_name']}, Page: {row['page_number']}, "
                    f"True: {row['true_label']}, Pred: {row['predicted_label']}, Text: {row['text']}\n")

    print(f"Logged {len(misclassified_rows)} misclassified rows to 'misclassified_rows.txt'.")


if __name__ == "__main__":
    model_output_dir = "bert_model"
    testing_data_csv = "testing_data.csv"

    if not os.path.isdir(model_output_dir):
        raise FileNotFoundError(f"The directory {model_output_dir} does not exist.")
    if not os.path.isfile(testing_data_csv):
        raise FileNotFoundError(f"The file {testing_data_csv} does not exist.")

    tokenizer = BertTokenizer.from_pretrained(model_output_dir)
    model = BertForSequenceClassification.from_pretrained(model_output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data = pd.read_csv(testing_data_csv)

    if 'text' not in data.columns or 'label' not in data.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    data['text'] = data['text'].astype(str).fillna("")

    texts = data['text'].tolist()
    labels = data['label'].tolist()

    predictions = []
    for text in texts:
        if not isinstance(text, str) or text.strip() == "":
            print(f"Skipping invalid text: {text}")
            continue
        probs = predict(text, model, tokenizer, device=device).squeeze(0)
        predictions.append(probs.cpu().numpy())

    predictions = np.array(predictions)
    evaluate_model(predictions, labels, texts, data['file_name'], data['page_number'])