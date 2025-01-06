from transformers import BertTokenizer, BertModel
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn
import os

# Custom model class for multi-task learning
class MultiTaskBertModel(nn.Module):
    def __init__(self, base_model, num_types):
        super(MultiTaskBertModel, self).__init__()
        self.base_model = base_model
        self.first_page_classifier = nn.Linear(base_model.config.hidden_size, 2)  # Binary classifier
        self.type_classifier = nn.Linear(base_model.config.hidden_size, num_types)  # Multiclass classifier

    def forward(self, input_ids, attention_mask, labels_first_page=None, labels_type=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        logits_first_page = self.first_page_classifier(pooled_output)
        logits_type = self.type_classifier(pooled_output)

        loss = None
        if labels_first_page is not None and labels_type is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss_first_page = loss_fn(logits_first_page, labels_first_page)
            loss_type = loss_fn(logits_type, labels_type)
            loss = loss_first_page + loss_type

        return {
            'loss': loss,
            'logits_first_page': logits_first_page,
            'logits_type': logits_type
        }

# Function to preprocess data and create a Dataset object
def preprocess_data(csv_file, tokenizer, max_length=512):
    data = pd.read_csv(csv_file)

    # Validate that the necessary columns exist
    if 'text' not in data.columns or 'is_first_page' not in data.columns or 'type' not in data.columns:
        raise ValueError("CSV file must contain 'text', 'is_first_page', and 'type' columns.")

    # Ensure no missing values
    data = data.dropna(subset=['text', 'is_first_page', 'type']).reset_index(drop=True)

    data['text'] = data['text'].astype(str)  # Ensure text data is string
    data['is_first_page'] = data['is_first_page'].astype(int)  # Ensure labels are integers
    data['type'] = data['type'].astype(int)  # Ensure labels are integers

    texts = data['text'].tolist()
    labels_first_page = data['is_first_page'].tolist()
    labels_type = data['type'].tolist()

    # Tokenize the text data
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    # Convert to Dataset object
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels_first_page': torch.tensor(labels_first_page),
        'labels_type': torch.tensor(labels_type)
    })
    return dataset

# Main function to create and train a multi-task BERT model
if __name__ == "__main__":
    # Paths
    training_data_csv = "training_data.csv"  # Training data file
    testing_data_csv = "testing_data.csv"    # Testing data file
    model_output_dir = "multi_task_bert_model"  # Directory to save the trained model

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the tokenizer and base model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    base_model = BertModel.from_pretrained('bert-base-uncased')

    # Determine the number of unique types for the type classifier
    data = pd.read_csv(training_data_csv)
    num_types = data['type'].nunique()

    # Initialize the multi-task model
    model = MultiTaskBertModel(base_model, num_types=num_types)
    model.to(device)

    # Preprocess training and testing data
    print("Preprocessing training data...")
    train_dataset = preprocess_data(training_data_csv, tokenizer)

    print("Preprocessing testing data...")
    test_dataset = preprocess_data(testing_data_csv, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,  # Matches train batch size for consistency
        warmup_steps=500,
        weight_decay=0.01,  # Apply weight decay for regularization
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",  # Enables evaluation during training
        eval_steps=500,
        save_strategy="steps",  # Save checkpoint periodically
        save_steps=500,
        save_total_limit=3,  # Retain only the latest 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Save model based on evaluation loss
        greater_is_better=False,
        no_cuda=(device != "cuda"),
        fp16=True,  # Use mixed precision for faster training
        dataloader_num_workers=4,  # Speed up data loading
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    print("Training the model...")
    trainer.train()

    # Save the model
    print(f"Saving the model to {model_output_dir}...")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    # Save the model in .bin format
    bin_model_path = os.path.join(model_output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), bin_model_path)
    print(f"Model saved as {bin_model_path}")
