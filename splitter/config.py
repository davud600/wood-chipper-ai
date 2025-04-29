import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# default values if not specified as args.

use_fp16 = True
use_all_types = False

training_mini_batch_size = 3
testing_mini_batch_size = 3

lr_cnn = 0.01
wd_cnn = 0.00005
isolated_epochs_cnn = 0

lr_llm = 0.005
wd_llm = 0.0001
isolated_epochs_llm = 2

lr_mlp = 0.00005
wd_mlp = 0.000125

epochs = 1000
pw_multiplier = 1
patience = 13
factor = 0.5

log_steps = 20
eval_steps = 1000
