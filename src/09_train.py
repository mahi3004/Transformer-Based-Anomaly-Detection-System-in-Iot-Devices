import json
import numpy as np
from common import DATA_DIR
from train_utils import train_model

ROOT = DATA_DIR.parent
PROC = DATA_DIR / "processed"
MODELS = ROOT / "models";   MODELS.mkdir(exist_ok=True)
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

SMOKE_TEST = False     
EPOCHS = 3
BATCH = 128
LR = 1e-3

# TODO 1: load
X_train = np.load(PROC / "X_train_scaled.npy")
y_train = np.load(PROC / "y_train.npy")
X_val = np.load(PROC / "X_val_scaled.npy")
y_val = np.load(PROC / "y_val.npy")

# TODO 2: smoke test = tiny subset, one epoch, separate file names (so the real files are safe)
if SMOKE_TEST:
    X_train, y_train = X_train[:20_000], y_train[:20_000]
    X_val, y_val = X_val[:5_000], y_val[:5_000]
    EPOCHS = 1
    model_path = MODELS / "iot_transformer_smoke.pth"
    history_path = RESULTS / "history_smoke.json"
else:
    model_path = MODELS / "iot_transformer_multiclass.pth"
    history_path = RESULTS / "history.json"

# TODO 3-8: the loop lives in train_utils.train_model (zero_grad -> forward -> loss -> backward -> step)
model, history = train_model(X_train, y_train, X_val, y_val,
                             epochs=EPOCHS, batch=BATCH, lr=LR,
                             save_path=model_path, print_every=200)

# TODO 9: save history
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)
print("history saved to", history_path)
