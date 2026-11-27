import json
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from common import DATA_DIR, CLASS_NAMES
from model import IoTTransformer
from train_utils import predict

ROOT = DATA_DIR.parent
PROC = DATA_DIR / "processed"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

# TODO 1: test data
X_test = np.load(PROC / "X_test_scaled.npy")
y_test = np.load(PROC / "y_test.npy")

# TODO 2: model + saved weights
model = IoTTransformer()
model.load_state_dict(torch.load(ROOT / "models" / "iot_transformer_multiclass.pth",
                                 map_location="cpu"))
model.eval()

# TODO 3: predict everything (batch 256, no shuffling)
y_pred = predict(model, X_test, batch=256)

# TODO 4: metrics
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
cm = confusion_matrix(y_test, y_pred)
print(f"accuracy {acc:.4f} | precision {prec:.4f} | recall {rec:.4f} | F1 {f1:.4f}")
print("confusion matrix (rows = actual, columns = predicted):")
print(cm)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

# TODO 5: save
with open(RESULTS / "metrics.json", "w") as f:
    json.dump({"accuracy": float(acc), "precision": float(prec),
               "recall": float(rec), "f1": float(f1)}, f, indent=2)
np.save(RESULTS / "confusion_matrix.npy", cm)
print("saved results/metrics.json and results/confusion_matrix.npy")
