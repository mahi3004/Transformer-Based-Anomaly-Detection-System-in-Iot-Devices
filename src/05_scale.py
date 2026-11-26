import numpy as np
import pandas as pd
from common import DATA_DIR, get_traffic_files

PROC = DATA_DIR / "processed"
MODELS = DATA_DIR.parent / "models"
MODELS.mkdir(exist_ok=True)

# TODO 1: load
X_train = np.load(PROC / "X_train.npy")
X_val = np.load(PROC / "X_val.npy")
X_test = np.load(PROC / "X_test.npy")

# TODO 2: mean and std from TRAIN only (float64 for accuracy)
mean = X_train.mean(axis=0, dtype=np.float64)
std = X_train.std(axis=0, dtype=np.float64)

# TODO 3: constant columns (std == 0) -> divide by 1 instead
zero_cols = np.where(std == 0)[0]
columns = pd.read_csv(get_traffic_files()[0], nrows=0).columns
print("constant columns:", len(zero_cols), [columns[i] for i in zero_cols])
std_safe = np.where(std == 0, 1.0, std)


# TODO 4: scale in chunks (keeps memory low), result back to float32
def scale(X, chunk=100_000):
    out = np.empty(X.shape, dtype=np.float32)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = ((X[i:i + chunk] - mean) / std_safe).astype(np.float32)
    return out


Xs_train, Xs_val, Xs_test = scale(X_train), scale(X_val), scale(X_test)

# TODO 5: sanity checks
ok = std != 0
for name, Xs in [("train", Xs_train), ("val", Xs_val), ("test", Xs_test)]:
    m = Xs.mean(axis=0, dtype=np.float64)
    s = Xs.std(axis=0, dtype=np.float64)
    print(f"{name:5s}: max|col mean|={np.abs(m).max():.2e}  "
          f"col std min={s[ok].min():.4f} max={s[ok].max():.4f}")
print("largest |value| in scaled train:", float(np.abs(Xs_train).max()))

# TODO 6: save
np.save(PROC / "X_train_scaled.npy", Xs_train)
np.save(PROC / "X_val_scaled.npy", Xs_val)
np.save(PROC / "X_test_scaled.npy", Xs_test)
np.savez(MODELS / "scaler.npz", mean=mean, std=std)
print("saved scaled arrays and models/scaler.npz")
