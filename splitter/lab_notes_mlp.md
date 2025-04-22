# NOTES

okay, one huge improvement was skipping isolated branch model training (session 14)
and lowering pw weight multiplier much more.
now i'm messing around with the random distance probability distribution.
gonna test erlang dist.

currently training (on 4) branch models with distance feature
and tuned params.
also (on 5) without distance feature to branch models & tuned params.

# DATASET

```
if doc_type != 3:
    continue
max_files = 500
max_pages_per_doc = 5
num_augmented = 2
```

## MLP

### architecture 3 -> 1

#### session 2 is music to my eyes:

```
[MLP] [STEP 2250] Epoch 90 | Loss: 0.0002 | true: 0 | pred: 0.00 | lr: 0.0001
  Loss: 11.9009 | F1: 0.8913 | Acc: 0.9465 | Rec: 0.8367 | Prec: 0.9535
  Confusion Matrix:
[[136   2]
 [  8  41]]

  ✅ Saved model [MLP] (F1: 0.8913)
```

### architecture 3 -> 8 -> 1

```
self.fusion_mlp = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(8, 1),
)
```

#### lr 0.001 wd 0.001
