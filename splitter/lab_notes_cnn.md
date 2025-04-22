# NOTES

locally running lr finder for distanceless cnn.

# DATASET

```
if doc_type != 3:
    continue
max_files = 500
max_pages_per_doc = 5
num_augmented = 2
```

## CNN

### architecture conv -> 64 -> 32 -> 1

```
self.classifier = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(32, 1),
)
```

and also added this line & blurs the image before input:

```
x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
```

#### decreased lr 0.001 & wd 0.0001

```
[CNN] [STEP 200] Epoch 2 | Loss: 1.0262 | true: 0 | pred: 0.56
Loss: 0.8074 | F1: 0.5561 | Acc: 0.6549 | Rec: 0.8421 | Prec: 0.4152
Confusion Matrix:
[[422 293]
 [ 39 208]]

[CNN] [STEP 1000] Epoch 9 | Loss: 0.8890 | true: 1 | pred: 0.11
Loss: 0.6266 | F1: 0.7283 | Acc: 0.8472 | Rec: 0.7976 | Prec: 0.6701
Confusion Matrix:
[[618  97]
 [ 50 197]]

[CNN] [STEP 1200] Epoch 10 | Loss: 0.8852 | true: 0 | pred: 0.05
Loss: 0.6152 | F1: 0.7166 | Acc: 0.8389 | Rec: 0.7935 | Prec: 0.6533
Confusion Matrix:
[[611 104]
 [ 51 196]]
```

#### decreased lr 0.0025 & wd 0.0005

```
[CNN] [STEP 200] Epoch 2 | Loss: 0.4016 | true: 1 | pred: 0.99
Loss: 0.6646 | F1: 0.7061 | Acc: 0.8503 | Rec: 0.7004 | Prec: 0.7119
Confusion Matrix:
[[645  70]
 [ 74 173]]

[CNN] [STEP 400] Epoch 4 | Loss: 0.2656 | true: 1 | pred: 0.91
Loss: 0.5691 | F1: 0.6909 | Acc: 0.8056 | Rec: 0.8462 | Prec: 0.5838
Confusion Matrix:
[[566 149]
 [ 38 209]]

[CNN] [STEP 600] Epoch 5 | Loss: 0.5709 | true: 0 | pred: 0.68
Loss: 0.8633 | F1: 0.6780 | Acc: 0.8617 | Rec: 0.5668 | Prec: 0.8434
Confusion Matrix:
[[689  26]
 [107 140]]
```

#### decreased lr 0.005 & wd 0.001

```
[CNN] [STEP 200] Epoch 2 | Loss: 0.7114 | true: 0 | pred: 0.26
Loss: 0.8782 | F1: 0.5928 | Acc: 0.8015 | Rec: 0.5628 | Prec: 0.6261
Confusion Matrix:
[[632  83]
 [108 139]]

[CNN] [STEP 400] Epoch 4 | Loss: 1.2899 | true: 1 | pred: 0.57
Loss: 0.8546 | F1: 0.6301 | Acc: 0.8389 | Rec: 0.5344 | Prec: 0.7674
Confusion Matrix:
[[675  40]
 [115 132]]

[CNN] [STEP 600] Epoch 5 | Loss: 0.7158 | true: 0 | pred: 0.00
Loss: 0.6590 | F1: 0.7588 | Acc: 0.8857 | Rec: 0.7004 | Prec: 0.8278
Confusion Matrix:
[[679  36]
 [ 74 173]]
```

#### default...

```
=== Training Configuration ===

[CNN Optimizer]
  Learning Rate      : 0.025
  Weight Decay       : 0.0025
  Isolated Steps     : 10000

[General]
  Epochs             : 10
  Pos‑Weight         : 2.8574
  Train Batch Size   : 12
  Test  Batch Size   : 12

==============================

[CNN] [STEP 200] Epoch 2 | Loss: 1.2343 | true: 0 | pred: 0.67
Loss: 0.7947 | F1: 0.5734 | Acc: 0.6705 | Rec: 0.8623 | Prec: 0.4294
Confusion Matrix:
[[432 283]
 [ 34 213]]

[CNN] [STEP 400] Epoch 3 | Loss: 0.5827 | true: 1 | pred: 0.98
Loss: 1.2849 | F1: 0.5558 | Acc: 0.7807 | Rec: 0.5344 | Prec: 0.5789
Confusion Matrix:
[[619  96]
 [115 132]]

[CNN] [STEP 600] Epoch 4 | Loss: 0.6505 | true: 1 | pred: 0.57
Loss: 0.7970 | F1: 0.5369 | Acc: 0.6019 | Rec: 0.8988 | Prec: 0.3828
Confusion Matrix:
[[357 358]
 [ 25 222]]
```

### architecture conv -> 32 -> 3 -> 1

```
self.classifier = nn.Sequential(
    nn.Linear(self.flattened_dim, 32),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(32, prev_pages_to_append + 1 + pages_to_append),
    nn.ReLU(),
    nn.Linear(prev_pages_to_append + 1 + pages_to_append, 1),
)
```

#### lr 0.0001 and wd 0.0

```
[CNN] [STEP 200] Epoch 1 | Loss: 1.0192 | true: 0 | pred: 0.50
Loss: 1.0259 | F1: 0.4086 | Acc: 0.2568 | Rec: 1.0000 | Prec: 0.2568
Confusion Matrix:
[[  0 715]
 [  0 247]]
```

#### default...

```
[CNN] [STEP 200] Epoch 1 | Loss: 1.1832 | true: 0 | pred: 0.50
Loss: 1.0266 | F1: 0.0000 | Acc: 0.7432 | Rec: 0.0000 | Prec: 0.0000
Confusion Matrix:
[[715   0]
 [247   0]]
```
