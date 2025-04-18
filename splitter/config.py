import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

training_mini_batch_size = 64
testing_mini_batch_size = 64
learning_rate = 0.00005
weight_decay = 0.00001
patience = 15
factor = 0.5
epochs = 10
log_steps = 10
eval_steps = 100
llm_warmup_steps = 0
cnn_warmup_steps = 0
mlp_warmup_steps = 0
