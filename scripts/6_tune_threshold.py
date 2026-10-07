import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.config import CHECKPOINTS_DIR, TABLES_DIR
from src.data.dataset import get_dataset
from src.models.efficientnet_cbam import EfficientNetB0_CBAM

m = sys.argv[1] if len(sys.argv) > 1 else "conjunctiva"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
ck = torch.load(CHECKPOINTS_DIR / m / "best_model.pth", map_location="cpu", weights_only=False)
net = EfficientNetB0_CBAM(num_classes=2, pretrained=False)
net.load_state_dict(ck["model_state_dict"]); net.to(dev).eval()

def predict(split, tta):
    dl = DataLoader(get_dataset(m, split=split), batch_size=32, shuffle=False, num_workers=0)
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in dl:
            x = x.to(dev); p = torch.softmax(net(x), 1)[:, 1]
            if tta: p = (p + torch.softmax(net(torch.flip(x, [3])), 1)[:, 1]) / 2
            ps.append(p.cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)

rows = []
def report(name, y, p, t):
    yh = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    r = {"Versi": name, "Threshold": round(t, 3), "n": len(y), "Accuracy": (tp+tn)/len(y),
         "Sensitivity": tp/(tp+fn), "Specificity": tn/(tn+fp), "F1": f1_score(y, yh), "AUC": roc_auc_score(y, p)}
    rows.append(r)
    print(f"{name:40s} thr={t:.2f} | Acc {r['Accuracy']:.4f} | Sens {r['Sensitivity']:.4f} | Spec {r['Specificity']:.4f} | F1 {r['F1']:.4f} | AUC {r['AUC']:.4f}")

yv, pv = predict("val", tta=True)
fpr, tpr, thr = roc_curve(yv, pv); t_val = float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))
print(f"\n=== {m.upper()} (test set) ===")
y0, p0 = predict("test", tta=False); report("Awal (threshold 0.5, tanpa TTA)", y0, p0, 0.5)
y1, p1 = predict("test", tta=True);  report("TTA + threshold dari val", y1, p1, t_val)
if m == "conjunctiva":
    yc, pc = predict("crossdomain_external_test", tta=True)
    print(f"\n=== {m.upper()} (cross-domain) ===")
    report("Cross-domain: TTA + threshold dari val", yc, pc, t_val)
pd.DataFrame(rows).round(4).to_csv(TABLES_DIR / f"{m}_threshold_tta.csv", index=False)
