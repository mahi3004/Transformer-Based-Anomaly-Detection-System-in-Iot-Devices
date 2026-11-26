import numpy as np
from sklearn.model_selection import train_test_split
from common import DATA_DIR

PROC = DATA_DIR / "processed"
X = np.load(PROC / "X_balanced.npy")
y = np.load(PROC / "y_balanced.npy")
dev = np.load(PROC / "device_balanced.npy")

# split the INDICES; first the 15% test set
idx = np.arange(len(y))
idx_rest, idx_test = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)

# validation must be 15% of the ORIGINAL data -> 0.15 / 0.85 of the rest
idx_train, idx_val = train_test_split(idx_rest, test_size=0.15 / 0.85,
                                      stratify=y[idx_rest], random_state=42)

# sizes and class counts
for name, ids in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
    print(f"{name:5s}: {len(ids):>9,} rows, per class {np.bincount(y[ids])}")

# save everything
for name, ids in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
    np.save(PROC / f"X_{name}.npy", X[ids])
    np.save(PROC / f"y_{name}.npy", y[ids])
    np.save(PROC / f"dev_{name}.npy", dev[ids])
print("saved split files in", PROC)
