# Self-Pruning Neural Network on CIFAR-10

A feedforward neural network that learns to prune its own weights **during training** using learnable gate parameters and L1 sparsity regularization — implemented from scratch in PyTorch.

---

## What This Project Does

Traditional neural networks are pruned *after* training as a separate optimization step. In contrast, this project integrates pruning directly into the training process.

Each weight is paired with a learnable **gate parameter** that controls its importance. During training:

* Important weights retain high gate values
* Unimportant weights are pushed toward zero and effectively removed

This results in a **sparse, efficient model** with minimal accuracy loss.

---

## How It Works

### PrunableLinear Layer

A custom implementation of a linear layer with an additional parameter `gate_scores`.

Forward pass:

1. Apply sigmoid: `gates = sigmoid(gate_scores)` → values in [0, 1]
2. Element-wise multiply: `pruned_weights = weight * gates`
3. Perform standard linear transformation

This ensures gradients flow through both weights and gates, allowing the model to learn which connections to prune.

---

### Sparsity Loss

The total loss is:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

Where:

* `SparsityLoss` = sum of all gate values (L1 penalty)
* λ controls the strength of pruning

Higher λ → more aggressive pruning

---

### Why L1 Encourages Sparsity

L1 regularization applies a **constant gradient**, pushing values all the way to zero. This makes it ideal for pruning.

In contrast, L2 only shrinks values and rarely produces exact zeros.

---

### Important Note on Gating

Since sigmoid outputs are continuous, gate values never become exactly zero.
Therefore, pruning is approximated using a threshold:

```
gate < 1e-2 → treated as pruned
```

---

## Network Architecture

```
Input (3 × 32 × 32 = 3072)
       ↓
PrunableLinear(3072 → 512) + ReLU
       ↓
PrunableLinear(512 → 256) + ReLU
       ↓
PrunableLinear(256 → 10)
       ↓
Output (10 classes)
```

A simple MLP is intentionally used to isolate the effect of pruning without architectural complexity.

---

## Results

Trained on CIFAR-10 for 10 epochs using Adam (lr=0.001).
Sparsity measured as % of gates below threshold (1e-2).

| Lambda (λ)   | Test Accuracy | Sparsity |
| ------------ | ------------- | -------- |
| 0 (Baseline) | 55.50%        | 0%       |
| 1e-05        | 56.12%        | 1.92%    |
| 1e-04        | 55.82%        | 36.34%   |
| 1e-03        | 51.61%        | 94.67%   |

### Observations

* Increasing λ increases sparsity but reduces accuracy
* Very high λ leads to excessive pruning and model collapse
* Best trade-off observed at **λ = 1e-04**

---

## Gate Value Distribution

Histograms of gate values show:

* Low λ → wide distribution (minimal pruning)
* Medium λ → spike near zero + active weights (ideal)
* High λ → most gates collapse to zero (over-pruning)

---

## Limitations

* Sigmoid-based gating does not produce exact zeros
* High λ can overly prune and degrade performance
* MLP architecture limits achievable accuracy on CIFAR-10
* Threshold-based pruning is heuristic

---

## How to Run

1. Open notebook in Google Colab
2. Enable GPU 
3. Run all cells sequentially

Runtime: ~20–25 minutes

---

## Dependencies

* Python 3.x
* PyTorch
* torchvision
* matplotlib

---

## Key Takeaway

This project demonstrates how **model architecture can adapt dynamically during training**, learning not just weights but also which connections are necessary.

It highlights the trade-off between **model efficiency (sparsity)** and **performance (accuracy)** — a key challenge in real-world AI systems.
