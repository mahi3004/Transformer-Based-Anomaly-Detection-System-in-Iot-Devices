"""Shared training / prediction helpers (used by 09_train.py and 11_device_holdout.py)."""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import IoTTransformer


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                       torch.from_numpy(np.asarray(y)).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def validate(model, loader, criterion):
    """Average loss and accuracy over a loader (no dropout, no gradients)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def train_model(X_train, y_train, X_val, y_val, epochs=3, batch=128, lr=1e-3,
                save_path=None, print_every=200, seed=42):
    """Train a fresh IoTTransformer; save the best (lowest val loss) weights."""
    torch.manual_seed(seed)
    train_loader = make_loader(X_train, y_train, batch, shuffle=True)
    val_loader = make_loader(X_val, y_val, 256, shuffle=False)

    model = IoTTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history, best_val = [], float("inf")
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        running = 0.0
        for i, (xb, yb) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if i % print_every == 0:
                print(f"  epoch {epoch} batch {i}/{len(train_loader)} "
                      f"loss {loss.item():.4f}", flush=True)
        train_loss = running / len(train_loader)

        val_loss, val_acc = validate(model, val_loader, criterion)
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_acc": val_acc, "seconds": time.time() - start}
        history.append(rec)
        print(rec, flush=True)

        if val_loss < best_val:
            best_val = val_loss
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                print("  saved best model to", save_path, flush=True)
    return model, history


def predict(model, X, batch=256):
    """Predicted class numbers for every row of X."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch], dtype=np.float32))
            preds.append(model(xb).argmax(dim=1).numpy())
    return np.concatenate(preds)
