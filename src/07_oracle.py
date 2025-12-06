# oracle analysis: test if demographics help prediction
# expected results: -0.21% improvement, CI [-1.56%, +1.54%]
# outputs: Table 4

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

print("--- oracle analysis ---")

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

# train/test split
X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(X, y, demos, test_size=0.2, random_state=42)
d_tr = d_tr.reset_index(drop=True)
d_te = d_te.reset_index(drop=True)
print(f"train: {len(X_tr)}, test: {len(X_te)}")

print("\n--- baseline ---")
clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
clf.fit(X_tr, y_tr)
acc_global = np.mean(clf.predict(X_te) == y_te)
print(f"global accuracy: {acc_global:.4f}")

print("\n--- group models ---")
g_res = []

for demo in ['age', 'gender', 'region']:
    for g in d_tr[demo].unique():
        if pd.isna(g):
            continue

        tr_mask = d_tr[demo] == g
        te_mask = d_te[demo] == g

        n_tr, n_te = tr_mask.sum(), te_mask.sum()
        if n_tr < 100 or n_te < 30:
            continue

        # train group-specific model
        clf_g = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
        clf_g.fit(X_tr[tr_mask], y_tr[tr_mask])
        acc_g = np.mean(clf_g.predict(X_te[te_mask]) == y_te[te_mask])

        # global model on same subset
        acc_glob = np.mean(clf.predict(X_te[te_mask]) == y_te[te_mask])
        imp = acc_g - acc_glob

        g_res.append({
            'demo': demo, 'group': g, 'n_tr': n_tr, 'n_te': n_te,
            'acc_global': acc_glob, 'acc_specific': acc_g, 'improvement': imp
        })
        print(f"{demo}={g}: global={acc_glob:.4f}, specific={acc_g:.4f}, Δ={imp:+.4f}")

grp_df = pd.DataFrame(g_res)
mean_imp_grp = grp_df['improvement'].mean()
print(f"mean improvement: {mean_imp_grp:+.4f}")

print("\n--- demo features ---")
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
demo_tr = enc.fit_transform(d_tr[['age', 'gender', 'region']])
demo_te = enc.transform(d_te[['age', 'gender', 'region']])

X_tr_aug = np.hstack([X_tr, demo_tr])
X_te_aug = np.hstack([X_te, demo_te])

clf_demo = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
clf_demo.fit(X_tr_aug, y_tr)
acc_demo = np.mean(clf_demo.predict(X_te_aug) == y_te)

print(f"with demographics: {acc_demo:.4f}")
print(f"improvement: {acc_demo - acc_global:+.4f}")

print("\n--- thresholds ---")
X_tr2, X_val, y_tr2, y_val, d_tr2, d_val = train_test_split(
    X_tr, y_tr, d_tr, test_size=0.2, random_state=42
)
d_val = d_val.reset_index(drop=True)

clf_th = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
clf_th.fit(X_tr2, y_tr2)
probs_val = clf_th.predict_proba(X_val)[:, 1]
probs_te = clf_th.predict_proba(X_te)[:, 1]

t_res = []

for demo in ['age', 'gender', 'region']:
    for g in d_val[demo].unique():
        if pd.isna(g):
            continue

        val_mask = d_val[demo] == g
        te_mask = d_te[demo] == g

        if val_mask.sum() < 50 or te_mask.sum() < 30:
            continue

        # find optimal threshold on val
        best_t, best_acc = 0.5, 0
        for t in np.arange(0.3, 0.7, 0.01):
            acc = np.mean((probs_val[val_mask] > t) == y_val[val_mask])
            if acc > best_acc:
                best_acc, best_t = acc, t

        # evaluate on test
        acc_def = np.mean((probs_te[te_mask] > 0.5) == y_te[te_mask])
        acc_opt = np.mean((probs_te[te_mask] > best_t) == y_te[te_mask])
        imp = acc_opt - acc_def

        t_res.append({
            'demo': demo, 'group': g, 'thresh': best_t, 'n_te': te_mask.sum(),
            'acc_default': acc_def, 'acc_optimized': acc_opt, 'improvement': imp
        })
        print(f"{demo}={g}: t={best_t:.2f}, default={acc_def:.4f}, opt={acc_opt:.4f}, Δ={imp:+.4f}")

thresh_df = pd.DataFrame(t_res)
mean_imp_thresh = thresh_df['improvement'].mean() if len(thresh_df) > 0 else 0
print(f"mean improvement: {mean_imp_thresh:+.4f}")

# super slow
print("\n--- bootstrap ---")
np.random.seed(42)
n_boot = 500
boot_imps = []
for b in range(n_boot):
    if b % 100 == 0:
        print(f"  {b}/{n_boot}...")

    # resample
    idx = np.random.choice(len(X), len(X), replace=True)
    X_b, y_b, d_b = X[idx], y[idx], demos.iloc[idx].reset_index(drop=True)

    X_tr_b, X_te_b, y_tr_b, y_te_b, d_tr_b, d_te_b = train_test_split(
        X_b, y_b, d_b, test_size=0.2, random_state=b
    )
    d_tr_b = d_tr_b.reset_index(drop=True)
    d_te_b = d_te_b.reset_index(drop=True)

    # global model
    clf_b = LogisticRegression(penalty='l2', C=1.0, max_iter=500, random_state=42)
    clf_b.fit(X_tr_b, y_tr_b)

    # group-specific improvements
    imps = []
    for demo in ['age', 'gender', 'region']:
        for g in d_tr_b[demo].unique():
            if pd.isna(g):
                continue
            tr_mask = d_tr_b[demo] == g
            te_mask = d_te_b[demo] == g
            if tr_mask.sum() < 50 or te_mask.sum() < 20:
                continue
            try:
                clf_g = LogisticRegression(penalty='l2', C=1.0, max_iter=500, random_state=42)
                clf_g.fit(X_tr_b[tr_mask], y_tr_b[tr_mask])
                acc_g = np.mean(clf_g.predict(X_te_b[te_mask]) == y_te_b[te_mask])
                acc_glob = np.mean(clf_b.predict(X_te_b[te_mask]) == y_te_b[te_mask])
                imps.append(acc_g - acc_glob)
            except:
                pass

    if len(imps) > 0:
        boot_imps.append(np.mean(imps))

ci_lo, ci_hi = np.percentile(boot_imps, [2.5, 97.5])
print(f"mean improvement: {mean_imp_grp*100:+.2f}%")
print(f"95% CI: [{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]")

# summary
best_imp = max(mean_imp_grp, acc_demo - acc_global, mean_imp_thresh)
print(f"\nbaseline acc: {acc_global:.4f}")
print(f"best oracle improvement: {best_imp*100:+.2f}%")
print("conclusion: demographics don't help" if best_imp < 0.01 else "conclusion: marginal improvement")

# save summary
summary = pd.DataFrame([{
    'acc_global': acc_global,
    'acc_with_demo': acc_demo,
    'imp_demo_features': acc_demo - acc_global,
    'imp_group_models': mean_imp_grp,
    'imp_thresholds': mean_imp_thresh,
    'best_improvement': best_imp,
    'ci_lo': ci_lo,
    'ci_hi': ci_hi
}])
summary.to_csv('data/oracle_summary.csv', index=False)
grp_df.to_csv('data/oracle_group_specific.csv', index=False)
thresh_df.to_csv('data/oracle_thresholds.csv', index=False)
print("saved: data/oracle_summary.csv, data/oracle_group_specific.csv, data/oracle_thresholds.csv")

print("done")
