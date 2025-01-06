from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch

# Function to preprocess data and create a Dataset object
def preprocess_data(csv_file, tokenizer, max_length=512):
    data = pd.read_csv(csv_file)

    if 'text' not in data.columns or 'label' not in data.columns:
        raise ValueError("CSV file must contain 'text' and 'label' columns.")

    if 'file_name' in data.columns and 'page_number' in data.columns:
        metadata = data[['file_name', 'page_number']]
    else:
        metadata = None

    data = data.dropna(subset=['text', 'label']).reset_index(drop=True)  # Reset index to align with metadata
    if metadata is not None:
        metadata = metadata.loc[data.index].reset_index(drop=True)  # Align metadata with filtered data

    data['text'] = data['text'].astype(str)
    data['label'] = data['label'].astype(int)

    texts = data['text'].tolist()
    labels = data['label'].tolist()

    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': torch.tensor(labels)  # Convert labels to tensor
    })

    if metadata is not None:
        dataset = dataset.add_column('metadata', metadata.to_dict('records'))

    return dataset

if __name__ == "__main__":
    training_data_csv = "training_data.csv"
    testing_data_csv = "testing_data.csv"
    model_output_dir = "bert_model"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    model.to(device)

    print("Preprocessing training data...")
    train_dataset = preprocess_data(training_data_csv, tokenizer)

    print("Preprocessing testing data...")
    test_dataset = preprocess_data(testing_data_csv, tokenizer)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        no_cuda=False,
        fp16=True,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    print("Training the model...")
    trainer.train()

    print(f"Saving the model to {model_output_dir}...")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    print("Model training and saving completed.")
