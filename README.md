# Self-Pruning Neural Network on CIFAR-10

A feedforward neural network that learns to prune its own weights during training using learnable gate parameters and L1 sparsity regularization — implemented from scratch in PyTorch.

---

## What This Project Does

Standard neural networks are pruned *after* training as a separate step. This project takes a different approach — the network learns *during* training which weights are unnecessary and removes them on its own.

Each weight in the network is paired with a learnable **gate** parameter. During training, a sparsity penalty pushes weak gates toward zero. A gate at zero means that weight is effectively dead — pruned. The result is a sparse network that is smaller and faster, with minimal loss in accuracy.

---

## How It Works

### PrunableLinear Layer
A custom replacement for `torch.nn.Linear`. In addition to the standard `weight` and `bias`, it has a `gate_scores` parameter of the same shape as the weights. During the forward pass:

1. `gate_scores` are passed through a **Sigmoid** function to produce `gates` in range [0, 1]
2. Weights are multiplied element-wise with gates: `pruned_weights = weight * gates`
3. The standard linear operation is applied using `pruned_weights`

Gradients flow through both `weight` and `gate_scores`, so the optimizer can learn to kill gates.

### Sparsity Loss
Training uses a combined loss:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

`SparsityLoss` is the **L1 norm** of all gate values — the sum of all sigmoid outputs across every `PrunableLinear` layer. Since L1 applies a constant gradient, it pushes gate values all the way to zero (unlike L2 which only shrinks them). A higher λ means stronger pruning pressure.

### Network Architecture

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

---

## Results

Trained on CIFAR-10 for 10 epochs with Adam optimizer (lr=0.001). Sparsity is measured as the percentage of gates below threshold 1e-2.

| Lambda (λ) | Test Accuracy | Sparsity Level |
|------------|--------------|----------------|
| 1e-05 | 56.39% | 2.16% |
| 1e-04 | 55.49% | 35.55% |
| 1e-03 | 51.57% | 94.78% |

As λ increases, the sparsity penalty gets stronger, forcing more gates to zero. Accuracy drops as a trade-off. The best balance is at **λ=1e-04** — 35% of weights pruned with only ~1% accuracy loss compared to the unpruned baseline.

---

## Why L1 Encourages Sparsity

The L1 penalty applies a **constant gradient** of ±1 to every gate value regardless of its magnitude. This means even very small gate values keep getting pushed toward zero until they actually reach zero. 

L2 by contrast applies a gradient proportional to the value itself — so as a gate shrinks, the push gets weaker and it never quite reaches zero.

L1 is therefore the natural choice when you want exact zeros, not just small values.

---

## Gate Value Distribution

The histograms below show the distribution of all gate values after training for each λ. A successful pruning shows a large spike near 0 (dead gates) and a smaller cluster of active gates away from 0.

**λ=1e-05** — Gates spread across [0, 0.6], low pruning pressure, network mostly intact

**λ=1e-04** — Large spike near 0, small tail of active gates, good sparsity-accuracy balance

**λ=1e-03** — Massive spike at 0, almost all gates dead, aggressive pruning

---

## How to Run

Open the notebook in Google Colab with a T4 GPU runtime.

Run all cells in order:
1. Imports and `PrunableLinear` definition
2. `SelfPruningNet` definition
3. Dataset loading and `train()` function
4. `evaluate()` function
5. `plot_gates()` function
6. Final training, evaluation, and plotting for all three λ values

Total runtime is approximately 20–25 minutes on a T4 GPU.

---

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- matplotlib

All available by default in Google Colab.
