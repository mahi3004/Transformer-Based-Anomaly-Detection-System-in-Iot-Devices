import sys
import numpy as np
import pandas as pd
from common import DATA_DIR, LABELS, parse_name, get_traffic_files

OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(exist_ok=True)

TARGET = 555_932                                          # = number of Benign rows
CLASS_TOTALS = {0: 555_932, 1: 3_668_402, 2: 2_838_272}   # from Stage 2
rng = np.random.default_rng(42)                           # fixed seed

# TODO 1: fraction of each class to keep (1% safety margin, never above 1.0)
keep_fraction = {c: min(1.0, TARGET / CLASS_TOTALS[c] * 1.01) for c in CLASS_TOTALS}
print("keep fractions:", keep_fraction)

X_parts = {0: [], 1: [], 2: []}
dev_parts = {0: [], 1: [], 2: []}

for path in get_traffic_files():
    device, class_name = parse_name(path)
    label = LABELS[class_name]
    df = pd.read_csv(path)
    # TODO 2: random fraction of this file's rows (different but reproducible seed per file)
    kept = df.sample(frac=keep_fraction[label],
                     random_state=int(rng.integers(0, 2**31 - 1)))
    # TODO 3: float32 numbers
    X_parts[label].append(kept.to_numpy(dtype=np.float32))
    # TODO 4: device number for every kept row
    dev_parts[label].append(np.full(len(kept), device, dtype=np.int8))
    print("done", path.name)

X_cls, dev_cls = {}, {}
for c in (0, 1, 2):
    # TODO 5: stack the parts of this class
    Xc = np.concatenate(X_parts[c])
    dc = np.concatenate(dev_parts[c])
    X_parts[c] = dev_parts[c] = None            # free memory

    # TODO 6: trim to exactly TARGET (or stop if too few)
    if len(Xc) < TARGET:
        sys.exit(f"ERROR: class {c} has only {len(Xc)} rows, need {TARGET}. "
                 f"Increase the safety margin in TODO 1.")
    if len(Xc) > TARGET:
        idx = rng.choice(len(Xc), size=TARGET, replace=False)
        Xc, dc = Xc[idx], dc[idx]
    X_cls[c], dev_cls[c] = Xc, dc

# TODO 7: combine the three classes
X = np.concatenate([X_cls[0], X_cls[1], X_cls[2]])
device = np.concatenate([dev_cls[0], dev_cls[1], dev_cls[2]])
y = np.concatenate([np.full(TARGET, c, dtype=np.int64) for c in (0, 1, 2)])
del X_cls, dev_cls

# TODO 8: shuffle X, y and device together with ONE permutation
perm = rng.permutation(len(y))
X, y, device = X[perm], y[perm], device[perm]

# TODO 9: report and save
print("X", X.shape, "y", y.shape, "device", device.shape)
print("count per class:", np.bincount(y))
np.save(OUT_DIR / "X_balanced.npy", X)
np.save(OUT_DIR / "y_balanced.npy", y)
np.save(OUT_DIR / "device_balanced.npy", device)
print("saved to", OUT_DIR)
