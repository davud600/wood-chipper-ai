import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

training_mini_batch_size = 20
testing_mini_batch_size = 20
learning_rate = 0.000075
weight_decay = 0.0005
patience = 15
factor = 0.5
epochs = 25
log_steps = 50
eval_steps = 200
cnn_warmup_steps = 200
llm_warmup_steps = 200
