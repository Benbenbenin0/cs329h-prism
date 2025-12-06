# calibration: ECE analysis with temperature/Platt scaling
# expected results: ECE = 0.017, T = 0.94
# outputs: Figure 3

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.special import expit as sigmoid
from scipy.optimize import minimize_scalar, minimize
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("--- calibration ---")

pairs = pd.read_csv('data/pairs.csv')
print(f"pairs: {len(pairs)}")

print("--- loading model ---")
model = SentenceTransformer('all-mpnet-base-v2')
resps = list(set(pairs['response_a'].unique()) | set(pairs['response_b'].unique()))
print(f"--- encoding {len(resps)} responses ---")
embs = model.encode(resps, show_progress_bar=True, batch_size=64)
emb_lookup = {r: e for r, e in zip(resps, embs)}

# create features
X, y, demos = [], [], []
for _, r in pairs.iterrows():
    X.append(emb_lookup[r['response_a']] - emb_lookup[r['response_b']])
    y.append(r['preference'])
    demos.append({'age': r['age'], 'gender': r['gender'], 'region': r['region']})

X, y = np.array(X), np.array(y)
demos = pd.DataFrame(demos)

# train/val/test split
X_tv, X_te, y_tv, y_te, d_tv, d_te = train_test_split(X, y, demos, test_size=0.2, random_state=42)
X_tr, X_va, y_tr, y_va, d_tr, d_va = train_test_split(X_tv, y_tv, d_tv, test_size=0.25, random_state=42)
print(f"train: {len(X_tr)}, val: {len(X_va)}, test: {len(X_te)}")

# train model
clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
clf.fit(X_tr, y_tr)

logits_va = clf.decision_function(X_va)
logits_te = clf.decision_function(X_te)
probs_te = clf.predict_proba(X_te)[:, 1]

acc = np.mean((probs_te > 0.5) == y_te)
print(f"test accuracy: {acc:.3f}")

# ECE calculation
def calc_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error (equal-width bins)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        n = mask.sum()
        if n > 0:
            acc_bin = np.mean(labels[mask])
            conf_bin = np.mean(probs[mask])
            ece += (n / len(probs)) * abs(acc_bin - conf_bin)
            bin_data.append({'bin': i, 'n': n, 'acc': acc_bin, 'conf': conf_bin})
    return ece, pd.DataFrame(bin_data)

ece_raw, bins_df = calc_ece(probs_te, y_te)
print(f"\nECE (uncalibrated): {ece_raw:.4f}")

# temperature scaling
def temp_nll(T, logits, labels):
    """Negative log-likelihood for temperature scaling optimization."""
    if T <= 0:
        return np.inf
    p = np.clip(sigmoid(logits / T), 1e-10, 1 - 1e-10)
    return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

res = minimize_scalar(temp_nll, bounds=(0.1, 10), args=(logits_va, y_va), method='bounded')
T = res.x
probs_temp = sigmoid(logits_te / T)
ece_temp, _ = calc_ece(probs_temp, y_te)
print(f"temperature T: {T:.2f}")
print(f"ECE (temperature scaled): {ece_temp:.4f}")

# Platt scaling
def platt_nll(params, logits, labels):
    """Negative log-likelihood for Platt scaling (a*logits + b)."""
    a, b = params
    p = np.clip(sigmoid(a * logits + b), 1e-10, 1 - 1e-10)
    return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

res = minimize(platt_nll, x0=[1.0, 0.0], args=(logits_va, y_va), method='L-BFGS-B')
a, b = res.x
probs_platt = sigmoid(a * logits_te + b)
ece_platt, _ = calc_ece(probs_platt, y_te)
print(f"Platt params: a={a:.3f}, b={b:.3f}")
print(f"ECE (Platt scaled): {ece_platt:.4f}")

print("\n--- by group ---")
d_te = d_te.reset_index(drop=True)
cal_res = []

for demo in ['age', 'gender', 'region']:
    for g in d_te[demo].unique():
        if pd.isna(g):
            continue
        mask = d_te[demo] == g
        if mask.sum() < 50:
            continue
        ece_g, _ = calc_ece(probs_te[mask], y_te[mask])
        cal_res.append({'demo': demo, 'group': g, 'n': mask.sum(), 'ece': ece_g})
        print(f"{demo}={g}: ECE={ece_g:.4f}, n={mask.sum()}")

grp_df = pd.DataFrame(cal_res)

print("\n--- creating figure ---")
fig, ax = plt.subplots(figsize=(7, 6))

ax.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)
valid = bins_df[bins_df['n'] > 0]
ax.bar(valid['conf'], valid['acc'], width=0.08, alpha=0.7, label=f'Model (ECE={ece_raw:.3f})')

ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Actual Probability', fontsize=12)
ax.set_title(f'Calibration Curve (ECE = {ece_raw:.3f})', fontsize=14)
ax.legend()
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('figures/fig3_calibration.png', dpi=150, bbox_inches='tight')
print("saved: figures/fig3_calibration.png")

# save results
pd.DataFrame([{
    'ece_raw': ece_raw, 'ece_temp': ece_temp, 'ece_platt': ece_platt,
    'T': T, 'a': a, 'b': b, 'test_acc': acc, 'n_test': len(y_te)
}]).to_csv('data/calibration_results.csv', index=False)
grp_df.to_csv('data/calibration_by_group.csv', index=False)
print("saved: data/calibration_results.csv, data/calibration_by_group.csv")

print("done")
