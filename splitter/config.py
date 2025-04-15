import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_mini_batch_size = 20
testing_mini_batch_size = 20
learning_rate = 0.000075
weight_decay = 0.0005
patience = 15
factor = 0.5
epochs = 15
log_steps = 50
eval_steps = 100
cnn_warmup_steps = 100
llm_warmup_steps = 100
