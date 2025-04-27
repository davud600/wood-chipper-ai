import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# default values if not specified as args.

train_in_fp16 = True
train_with_all_data = True

training_mini_batch_size = 64
testing_mini_batch_size = 64

lr_cnn = 0.01
wd_cnn = 0.0001
isolated_epochs_cnn = 0

lr_llm = 0.01
wd_llm = 0.0001
isolated_epochs_llm = 0

lr_mlp = 0.00005
wd_mlp = 0.00025

epochs = 1000
pw_multiplier = 1
patience = 13
factor = 0.5

log_steps = 20
eval_steps = 1000
