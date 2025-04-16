import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

training_mini_batch_size = 48
testing_mini_batch_size = 48
learning_rate = 0.000075
weight_decay = 0.001
patience = 15
factor = 0.5
epochs = 25
log_steps = 50
eval_steps = 100
llm_warmup_steps = 100
cnn_warmup_steps = 100
