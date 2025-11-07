# Essentials Pytorch: From Basic Tensors to SVMs, Neural Networks, and Variational Autoencoders


Below is a **self‑contained PyTorch tutorial** that walks you through

1. **Core PyTorch concepts** – tensors, datatypes, basic ops, autograd, DataLoaders.  
2. **Training a linear SVM** (implemented with a hinge‑loss).  
3. **Training a simple feed‑forward neural network** on a toy classification task.  
4. **Building & training a Variational Auto‑Encoder (VAE)** on MNIST.

You can copy‑paste each section into a Jupyter notebook (or a `.py` file) and run it step‑by‑step.  
All code is written for **PyTorch ≥ 2.0** and works on CPU or GPU.

---

## 1️⃣  PyTorch Basics

### 1.1 Install & Imports
```bash
# In a terminal (or a notebook cell with !):
pip install torch torchvision tqdm matplotlib
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm   # nice progress bars
import numpy as np
```

### 1.2 Tensors & Datatypes
```python
# Create a 2‑D tensor (float32 by default)
x = torch.tensor([[1., 2.],
                  [3., 4.]])
print(x)

# Different dtypes
x_int = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)   # long/int64
x_float = x_int.to(torch.float32)                         # cast
x_bool = x > 2                                            # bool tensor

print("int dtype:", x_int.dtype)
print("float dtype:", x_float.dtype)
print("bool tensor:\n", x_bool)
```

### 1.3 Basic Operations (CPU ↔ GPU)
```python
# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)

# Arithmetic
y = torch.randn_like(x)          # same shape, random values
z = x + y                        # element‑wise add
z = torch.matmul(x, y.T)         # matrix multiplication

# Reductions
mean = z.mean()
max_val, max_idx = z.max(dim=1)   # per‑row max

print("mean:", mean.item())
print("row‑wise max:", max_val, max_idx)
```

### 1.4 Autograd – automatic differentiation
```python
# Any tensor with `requires_grad=True` tracks ops for back‑prop
w = torch.randn(2, 2, requires_grad=True, device=device)
b = torch.randn(2, requires_grad=True, device=device)

# Simple linear model
def linear(x):
    return x @ w + b

out = linear(x)               # forward pass
loss = out.pow(2).mean()      # dummy loss (MSE)

loss.backward()               # compute gradients
print("∂loss/∂w:\n", w.grad)
print("∂loss/∂b:\n", b.grad)

# Zero grads before next step
w.grad.zero_()
b.grad.zero_()
```

### 1.5 Datasets & DataLoaders
```python
# Synthetic 2‑class data (make_moons)
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

# Train/val split
train_dataset = TensorDataset(X, y)
train_len = int(0.8 * len(train_dataset))
val_len   = len(train_dataset) - train_len
train_set, val_set = random_split(train_dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=64)
```

---

## 2️⃣  Linear SVM with PyTorch

A classic SVM solves  

\[\min_{w,b}\ \frac{1}{2}\|w\|^2 + C\sum_i \max(0, 1 - y_i (w^\top x_i + b))\]


**Symbols in the formulation**

* `x_i` – feature vector of the i‑th training example (a column of the data matrix).  
* `y_i` – class label of the i‑th example, taking values **+1** or **‑1**.  
* `w` – weight vector that defines the orientation of the separating hyper‑plane.  
* `b` – bias (intercept) term that shifts the hyper‑plane away from the origin.  
* `C` – regularisation parameter that controls the trade‑off between a large margin (small `‖w‖`) and the amount of training error tolerated.  
* `‖w‖^2` – squared Euclidean norm of `w`; its half appears as the regularisation term.  
* `max(0, 1 – y_i (wᵀ x_i + b))` – the **hinge loss** for a single example. It is zero when the example is correctly classified with a margin of at least 1, and grows linearly when the margin is smaller.

**What the SVM optimisation does**

The objective  

\[
min_{w,b}  (1/2) * ‖w‖^2  +  C * Σ_i hingeloss_i
\]

has two competing goals:

1. **Maximise the margin** – minimizing `(1/2)‖w‖^2` makes the distance between the two class‑separating hyper‑planes as large as possible. A larger margin usually yields better generalisation.

2. **Control classification errors** – the sum of hinge losses penalises points that lie on the wrong side of the margin (or are mis‑classified). The constant `C` scales this penalty: a large `C` forces the optimizer to fit the training data more tightly (few violations, possibly over‑fitting), while a small `C` allows more violations in favour of a wider margin (more regularisation).

The solution `(w*, b*)` defines a linear decision rule:

```
predict(x) = sign( w*ᵀ x + b* )
```

If `w*ᵀ x + b*` is positive the model predicts `+1`; otherwise it predicts `‑1`.  

**Why implement hinge loss and use SGD?**

* The hinge loss is piecewise linear and convex, which makes it suitable for stochastic gradient descent (SGD).  
* SGD updates `w` and `b` using a single (or a small batch of) training examples at each step, allowing the SVM to be trained on large datasets where a full quadratic‑programming solution would be too costly.  

In practice, each SGD step computes the sub‑gradient of the hinge loss for the current example, updates `w` and `b` accordingly, and repeats over many epochs until the objective stabilises.

We’ll implement the **hinge loss** ourselves and use SGD.

### 2.1 Model & Hinge Loss
```python
class LinearSVM(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)   # output is a single logit

    def forward(self, x):
        return self.fc(x)                     # raw score (no sigmoid)

def hinge_loss(outputs, targets, margin=1.0):
    """
    outputs : (N, 1) raw scores
    targets : (N,) 0/1 -> convert to -1/+1
    """
    targets = targets.float().view(-1, 1) * 2 - 1   # 0→-1, 1→+1
    loss = torch.clamp(margin - outputs * targets, min=0)
    return loss.mean()
```

### 2.2 Training Loop
```python
model = LinearSVM(in_features=2).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)  # L2 = weight_decay

def train_one_epoch(loader):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = hinge_loss(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def evaluate(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = (logits.squeeze() > 0).long()
            correct += (preds == yb).sum().item()
    return correct / len(loader.dataset)

# Run training
epochs = 30
for epoch in range(1, epochs+1):
    train_loss = train_one_epoch(train_loader)
    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | val‑acc {val_acc:.3f}")
```

### 2.3 Visualising the Decision Boundary
```python
def plot_decision_boundary(model, X, y):
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200),
                         np.linspace(-1.0, 1.5, 200))
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
    model.eval()
    with torch.no_grad():
        scores = model(grid).cpu().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, scores, levels=[-float('inf'), 0, float('inf')],
                 alpha=0.3, colors=['#ff9999','#99ff99'])
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k')
    plt.title('Linear SVM decision boundary')
    plt.show()

plot_decision_boundary(model, X.numpy(), y.numpy())
```

> **Result:** you should see a straight line separating the two moon clusters (the SVM isn’t perfect because the data isn’t linearly separable, but it illustrates the workflow).

---

## 3️⃣  Simple Feed‑Forward Neural Network

Now we’ll train a **2‑layer MLP** on the same moon data, but with a non‑linear hidden layer.

### 3.1 Model Definition
```python
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)   # logits for 2 classes
        )

    def forward(self, x):
        return self.net(x)
```

### 3.2 Training Utilities
```python
criterion = nn.CrossEntropyLoss()          # combines LogSoftmax + NLLLoss
optimizer = torch.optim.Adam(MLP().parameters(), lr=1e-3)

def train_epoch(model, loader):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def accuracy(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
    return correct / len(loader.dataset)
```

### 3.3 Train the MLP
```python
mlp = MLP().to(device)
epochs = 200
train_losses, val_accs = [], []

for epoch in range(1, epochs+1):
    loss = train_epoch(mlp, train_loader)
    acc  = accuracy(mlp, val_loader)
    train_losses.append(loss)
    val_accs.append(acc)
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | loss {loss:.4f} | val‑acc {acc:.3f}")

# Plot learning curve
plt.plot(train_losses, label='train loss')
plt.plot(val_accs, label='val accuracy')
plt.legend()
plt.show()
```

### 3.4 Decision Boundary (non‑linear)
```python
def plot_mlp_boundary(model, X, y):
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 300),
                         np.linspace(-1.0, 1.5, 300))
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
    model.eval()
    with torch.no_grad():
        logits = model(grid)
        probs = logits.softmax(dim=1)[:,1].cpu().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, probs, levels=50, cmap='RdBu', alpha=0.6)
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k')
    plt.title('MLP decision boundary (probability of class 1)')
    plt.show()

plot_mlp_boundary(mlp, X.numpy(), y.numpy())
```

> **Observation:** the MLP can carve a curved boundary that separates the moons far better than the linear SVM.

---

## 4️⃣  Variational Auto‑Encoder (VAE)

A VAE learns a **probabilistic latent space** and can generate new data.  
We’ll train a VAE on the **MNIST** digit dataset (28 × 28 grayscale images).

### 4.1 Load MNIST
```python
transform = transforms.Compose([
    transforms.ToTensor(),                     # → [0,1] float tensor
    transforms.Normalize((0.5,), (0.5,))       # → [-1,1] for nicer training
])

mnist_train = torchvision.datasets.MNIST(root='.', train=True,
                                         download=True, transform=transform)
mnist_test  = torchvision.datasets.MNIST(root='.', train=False,
                                         download=True, transform=transform)

batch_size = 128
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader  = DataLoader(mnist_test,  batch_size=batch_size, shuffle=False, pin_memory=True)
```

### 4.2 VAE Architecture

```python
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        # Encoder (28x28 → 256 → latent)
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.mu_head   = nn.Linear(256, latent_dim)   # μ(z|x)
        self.logvar_head = nn.Linear(256, latent_dim) # log σ²

        # Decoder (latent → 256 → 28x28)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()          # because we normalized inputs to [-1,1]
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)          # σ = exp(logσ² / 2)
        eps = torch.randn_like(std)         # ε ~ N(0,1)
        return mu + eps*std                  # z = μ + σ·ε

    def decode(self, z):
        out = self.dec(z)
        return out.view(-1, 1, 28, 28)       # reshape to image

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
```

### 4.3 Loss Function

VAE loss = **Reconstruction loss** + **KL‑divergence**  

* Reconstruction: binary cross‑entropy (or MSE) between input and output.  
* KL term: forces the approximate posterior `q(z|x)` to stay close to a standard normal prior.

```python
def vae_loss(recon_x, x, mu, logvar):
    # BCE works well for normalized [-1,1] if we first map back to [0,1]
    recon_x = (recon_x + 1) / 2          # [-1,1] → [0,1]
    x = (x + 1) / 2

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence between N(μ,σ²) and N(0,1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

### 4.4 Training Loop
```python
latent_dim = 20
vae = VAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

def train_vae(epoch):
    vae.train()
    train_loss = 0
    for xb, _ in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
        xb = xb.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = vae(xb)
        loss = vae_loss(recon, xb, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch {epoch:02d} | avg loss {avg_loss:.4f}")

def test_vae():
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            recon, mu, logvar = vae(xb)
            loss = vae_loss(recon, xb, mu, logvar)
            test_loss += loss.item()
    return test_loss / len(test_loader.dataset)

epochs = 15
for epoch in range(1, epochs+1):
    train_vae(epoch)
    val = test_vae()
    print(f"   Validation loss: {val:.4f}")
```

### 4.5 Visualising Reconstructions & Sampling

```python
def show_images(imgs, nrow=8, title=''):
    imgs = imgs.cpu()
    grid = torchvision.utils.make_grid(imgs, nrow=nrow, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(10,4))
    plt.title(title)
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.show()

# 1️⃣ Reconstructions on test set
vae.eval()
with torch.no_grad():
    xb, _ = next(iter(test_loader))
    xb = xb.to(device)
    recon, _, _ = vae(xb)
show_images(xb[:64], title='Original test images')
show_images(recon[:64], title='VAE reconstructions')

# 2️⃣ Sampling from the prior
with torch.no_grad():
    z = torch.randn(64, latent_dim).to(device)   # N(0, I)
    samples = vae.decode(z)
show_images(samples, title='Samples generated from N(0, I)')
```

You should see that the VAE can **reconstruct** digits reasonably well and that the **samples** look like plausible handwritten digits, albeit a bit blurry (that's the classic VAE trade‑off).

---

## 🎯  Recap & Next Steps

| Section | What you learned | Key PyTorch APIs |
|--------|------------------|------------------|
| **1 – Basics** | Tensors, dtypes, device handling, autograd, DataLoaders | `torch.tensor`, `.to()`, `requires_grad`, `DataLoader` |
| **2 – Linear SVM** | Implement hinge loss, train a linear classifier with SGD | `nn.Linear`, custom loss, `optimizer.zero_grad()` |
| **3 – MLP** | Build a multi‑layer perceptron, use `CrossEntropyLoss`, monitor accuracy | `nn.Sequential`, `nn.ReLU`, `torch.optim.Adam` |
| **4 – VAE** | Encoder/decoder, reparameterization trick, KL‑divergence, generative sampling | `nn.Module`, `torch.randn_like`, `binary_cross_entropy` |

### Where to go from here?

1. **Experiment with other datasets** – e.g., CIFAR‑10 for convolutional VAEs.  
2. **Add regularisation** – dropout, batch‑norm, weight decay.  
3. **Try more advanced generative models** – GANs, Normalizing Flows, Diffusion models.  
4. **Deploy** – export a trained model with `torch.jit.trace` or `torchscript` and serve it via TorchServe.

Happy coding! 🚀 If you hit any roadblocks or want to dive deeper into a specific topic, just let me know.