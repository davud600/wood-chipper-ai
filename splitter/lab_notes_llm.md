# DATASET

```
if doc_type != 3:
    continue
max_files = 500
max_pages_per_doc = 5
num_augmented = 2
```

## LLM

### architecture cls -> 32 -> 1

```
self.classifier = nn.Sequential(
    nn.Linear(self.config.hidden_size, 32),
    nn.GELU(),
    nn.Dropout(dropout), # 0.1
    nn.Linear(32, 1),
)
```

#### lr 0.001 wd 0.001

### architecture cls -> 32 -> 3 -> 1

```
self.classifier = nn.Sequential(
    nn.Linear(self.config.hidden_size, 32),
    nn.GELU(),
    nn.Dropout(dropout), # 0.1
    nn.Linear(32, prev_pages_to_append + 1 + pages_to_append),
    nn.ReLU(),
    nn.Linear(prev_pages_to_append + 1 + pages_to_append, 1),
)
```

#### undoing last step, going back to 0.001

#### increased less lr & wd for llm foundational model from 0.001 to 0.0025.

```
[LLM] [STEP 200] Epoch 4 | Loss: 1.0513 | true: 1 | pred: 0.50
Loss: 1.0000 | F1: 0.1176 | Acc: 0.7349 | Rec: 0.0688 | Prec: 0.4048
Confusion Matrix:
[[690  25]
 [230  17]]

[LLM] [STEP 400] Epoch 7 | Loss: 0.9140 | true: 0 | pred: 0.36
Loss: 0.9656 | F1: 0.1224 | Acc: 0.7318 | Rec: 0.0729 | Prec: 0.3830
Confusion Matrix:
[[686  29]
 [229  18]]

[LLM] [STEP 600] Epoch 10 | Loss: 1.0560 | true: 0 | pred: 0.35
Loss: 0.9455 | F1: 0.6080 | Acc: 0.7869 | Rec: 0.6437 | Prec: 0.5761
Confusion Matrix:
[[598 117]
 [ 88 159]]
```

#### increased lr & wd for llm foundational model from 0.001 to 0.01.

```
[LLM] [STEP 200] Epoch 4 | Loss: 1.0211 | true: 0 | pred: 0.51
Loss: 1.0131 | F1: 0.4148 | Acc: 0.3254 | Rec: 0.9312 | Prec: 0.2668
Confusion Matrix:
[[ 83 632]
 [ 17 230]]

[LLM] [STEP 400] Epoch 7 | Loss: 0.9576 | true: 0 | pred: 0.48
Loss: 0.9699 | F1: 0.5400 | Acc: 0.7308 | Rec: 0.6154 | Prec: 0.4810
Confusion Matrix:
[[551 164]
 [ 95 152]]

[LLM] [STEP 600] Epoch 10 | Loss: 1.0000 | true: 0 | pred: 0.54
Loss: 0.9219 | F1: 0.6256 | Acc: 0.7349 | Rec: 0.8623 | Prec: 0.4908
Confusion Matrix:
[[494 221]
 [ 34 213]]
```

#### decreased lr & wd for llm foundational model from 0.001 to 0.0001.

```
[LLM] [STEP 400] Epoch 7 | Loss: 0.9822 | true: 0 | pred: 0.42
Loss: 0.9764 | F1: 0.1895 | Acc: 0.7422 | Rec: 0.1174 | Prec: 0.4915
Confusion Matrix:
[[685  30]
 [218  29]]
```

#### increased pw back to 1.0 for now, weight init activated

```
for m in self.classifier.modules():
    if isinstance(m, nn.Linear):
        init_weights(m)
```

```
[LLM] [STEP 200] Epoch 4 | Loss: 0.9973 | true: 0 | pred: 0.52
Loss: 0.9916 | F1: 0.5544 | Acc: 0.6975 | Rec: 0.7328 | Prec: 0.4458
Confusion Matrix:
[[490 225]
 [ 66 181]]

[LLM] [STEP 400] Epoch 7 | Loss: 0.9349 | true: 0 | pred: 0.47
Loss: 0.9421 | F1: 0.6210 | Acc: 0.7069 | Rec: 0.9352 | Prec: 0.4648
Confusion Matrix:
[[449 266]
 [ 16 231]]

[LLM] [STEP 600] Epoch 10 | Loss: 0.9415 | true: 0 | pred: 0.47
Loss: 0.9028 | F1: 0.6352 | Acc: 0.7349 | Rec: 0.8988 | Prec: 0.4912
Confusion Matrix:
[[485 230]
 [ 25 222]]
```

#### increased pw (from 0.75*n0/n1 to 0.9*n0/n1)

```
[LLM] [STEP 200] Epoch 4 | Loss: 1.0003 | true: 0 | pred: 0.42
Loss: 0.9480 | F1: 0.0000 | Acc: 0.7370 | Rec: 0.0000 | Prec: 0.0000
Confusion Matrix:
[[709   6]
 [247   0]]

[LLM] [STEP 400] Epoch 7 | Loss: 0.9287 | true: 1 | pred: 0.50
Loss: 0.9004 | F1: 0.5354 | Acc: 0.8087 | Rec: 0.4291 | Prec: 0.7114
Confusion Matrix:
[[672  43]
 [141 106]]

[LLM] [STEP 600] Epoch 10 | Loss: 0.9434 | true: 0 | pred: 0.42
Loss: 0.8723 | F1: 0.5811 | Acc: 0.6913 | Rec: 0.8340 | Prec: 0.4459
Confusion Matrix:
[[459 256]
 [ 41 206]]
```

#### wd decreased back to 0.0005 & smaller pw (from 1.0*n0/n1 to 0.75*n0/n1)

too many false negatives..

```
[LLM] [STEP 200] Epoch 4 | Loss: 0.8819 | true: 0 | pred: 0.50
Loss: 0.8616 | F1: 0.0000 | Acc: 0.7401 | Rec: 0.0000 | Prec: 0.0000
Confusion Matrix:
[[712   3]
 [247   0]]

[LLM] [STEP 400] Epoch 7 | Loss: 0.8117 | true: 0 | pred: 0.36
Loss: 0.8381 | F1: 0.0458 | Acc: 0.7401 | Rec: 0.0243 | Prec: 0.4000
Confusion Matrix:
[[706   9]
 [241   6]]
```

#### wd increased (0.005)

too many false positives...

```
[LLM] [STEP 200] Epoch 4 | Loss: 1.0228 | true: 0 | pred: 0.55
Loss: 1.0012 | F1: 0.4238 | Acc: 0.3046 | Rec: 0.9960 | Prec: 0.2691
Confusion Matrix:
[[ 47 668]
 [  1 246]]
```

#### wd increased (0.0005)

```
[LLM] [STEP 200] Epoch 4 | Loss: 0.9969 | true: 1 | pred: 0.55
Loss: 0.9804 | F1: 0.6016 | Acc: 0.7453 | Rec: 0.7490 | Prec: 0.5027
Confusion Matrix:
[[532 183]
 [ 62 185]]

[LLM] [STEP 400] Epoch 7 | Loss: 0.8982 | true: 0 | pred: 0.41
Loss: 0.9369 | F1: 0.6427 | Acc: 0.7931 | Rec: 0.7247 | Prec: 0.5774
Confusion Matrix:
[[584 131]
 [ 68 179]]

[LLM] [STEP 600] Epoch 10 | Loss: 0.9870 | true: 0 | pred: 0.54
Loss: 0.8990 | F1: 0.6409 | Acc: 0.7588 | Rec: 0.8381 | Prec: 0.5188
Confusion Matrix:
[[523 192]
 [ 40 207]]
```

#### wd changed to non zero (0.00005)

doesn't seem to work...

```
[LLM] [STEP 200] Epoch 4 | Loss: 1.0195 | true: 0 | pred: 0.50
Loss: 1.0182 | F1: 0.0000 | Acc: 0.7432 | Rec: 0.0000 | Prec: 0.0000
Confusion Matrix:
[[715   0]
 [247   0]]
```

#### lr 0.00005

```
[training]
[INFO] Loaded 1930 valid rows (with existing images)
[INFO] First pages: 497, Non-first pages: 1433
[testing]
[INFO] Loaded 962 valid rows (with existing images)
[INFO] First pages: 247, Non-first pages: 715
[INFO] Starting training...
=== Training Configuration ===

[LLM Optimizer]
Learning Rate : 5e-05
Weight Decay : 0.0
Isolated Steps : 10000

[General]
Epochs : 10
Pos‑Weight : 2.8833
Train Batch Size : 32
Test Batch Size : 32

==============================
[LLM] [STEP 200] Epoch 4 | Loss: 0.9937 | true: 0 | pred: 0.45
Loss: 0.9753 | F1: 0.4914 | Acc: 0.7225 | Rec: 0.5223 | Prec: 0.4640
Confusion Matrix:
[[566 149]
 [118 129]]

[LLM] [STEP 400] Epoch 7 | Loss: 0.8445 | true: 0 | pred: 0.44
Loss: 0.9205 | F1: 0.6096 | Acc: 0.7536 | Rec: 0.7490 | Prec: 0.5139
Confusion Matrix:
[[540 175]
 [ 62 185]]

[LLM] [STEP 600] Epoch 10 | Loss: 0.9333 | true: 0 | pred: 0.52
Loss: 0.8712 | F1: 0.6273 | Acc: 0.7703 | Rec: 0.7530 | Prec: 0.5376
Confusion Matrix:
[[555 160]
 [ 61 186]]
```

#### lr 0.0001

```
[training]
[INFO] Loaded 1929 valid rows (with existing images)
[INFO] First pages: 499, Non-first pages: 1430
[testing]
[INFO] Loaded 962 valid rows (with existing images)
[INFO] First pages: 247, Non-first pages: 715
[INFO] Starting training...
=== Training Configuration ===

[LLM Optimizer]
Learning Rate : 0.0001
Weight Decay : 0.0
Isolated Steps : 10000

[General]
Epochs : 10
Pos‑Weight : 2.8657
Train Batch Size : 32
Test Batch Size : 32

==============================

[LLM] [STEP 200] Epoch 4 | Loss: 1.0014 | true: 0 | pred: 0.52
Loss: 0.9855 | F1: 0.5078 | Acc: 0.5062 | Rec: 0.9919 | Prec: 0.3412
Confusion Matrix:
[[242 473]
 [  2 245]]

[LLM] [STEP 400] Epoch 7 | Loss: 0.9229 | true: 0 | pred: 0.46
Loss: 0.9465 | F1: 0.5949 | Acc: 0.6674 | Rec: 0.9514 | Prec: 0.4328
Confusion Matrix:
[[407 308]
 [ 12 235]]

[LLM] [STEP 600] Epoch 10 | Loss: 0.9652 | true: 0 | pred: 0.44
Loss: 0.8973 | F1: 0.6450 | Acc: 0.7505 | Rec: 0.8826 | Prec: 0.5082
Confusion Matrix:
[[504 211]
 [ 29 218]]
```
