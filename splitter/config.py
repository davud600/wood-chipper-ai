import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# default values if not specified as args.

training_mini_batch_size = 32
testing_mini_batch_size = 32

lr_cnn = 0.001
wd_cnn = 0.0005
isolated_epochs_cnn = 5

lr_llm = 0.001
wd_llm = 0.0005
isolated_epochs_llm = 5

lr_mlp = 0.0001
wd_mlp = 0.0001

epochs = 10
pw_multiplier = 1
patience = 10
factor = 0.5

log_steps = 10
eval_steps = 250
