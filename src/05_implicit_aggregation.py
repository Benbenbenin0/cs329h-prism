# implicit aggregation: test if RM rankings match Borda count (DPL Theorem 3.1)
# expected results: tau = 0.762, p < 0.0001
# outputs: Figure 2

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import kendalltau, rankdata
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("--- implicit aggregation ---")

pairs = pd.read_csv('data/pairs.csv')
turn0 = pd.read_csv('data/turn0_pairs.csv')
print(f"pairs: {len(pairs)}, turn0: {len(turn0)}")

print("--- loading model ---")
model = SentenceTransformer('all-mpnet-base-v2')

resps = list(set(pairs['response_a'].unique()) | set(pairs['response_b'].unique()))
print(f"--- encoding {len(resps)} responses ---")
embs = model.encode(resps, show_progress_bar=True, batch_size=64)
emb_lookup = {r: e for r, e in zip(resps, embs)}

print("--- training ---")
X, y = [], []
for _, r in pairs.iterrows():
    X.append(emb_lookup[r['response_a']] - emb_lookup[r['response_b']])
    y.append(r['preference'])
X, y = np.array(X), np.array(y)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
clf.fit(X_tr, y_tr)

acc = accuracy_score(y_te, clf.predict(X_te))
print(f"test accuracy: {acc:.3f}")

# derive RM rankings
rm_scores = defaultdict(lambda: {'s': 0, 'n': 0})
for _, r in pairs.iterrows():
    feat = (emb_lookup[r['response_a']] - emb_lookup[r['response_b']]).reshape(1, -1)
    p = clf.predict_proba(feat)[0, 1]
    rm_scores[r['model_a']]['s'] += p
    rm_scores[r['model_a']]['n'] += 1
    rm_scores[r['model_b']]['s'] += (1 - p)
    rm_scores[r['model_b']]['n'] += 1

rm_rank = sorted([(m, d['s']/d['n']) for m, d in rm_scores.items()], key=lambda x: -x[1])

# borda from turn0
conv_counts = turn0.groupby('conversation_id').size()
conv_4mod = conv_counts[conv_counts == 6].index  # C(4,2) = 6 pairs

borda = defaultdict(float)
n_rank = 0

for cid in conv_4mod:
    cdf = turn0[turn0['conversation_id'] == cid]
    scores = {}
    for _, r in cdf.iterrows():
        scores[r['model_a']] = r['score_a']
        scores[r['model_b']] = r['score_b']
    if len(scores) != 4:
        continue

    items = sorted(scores.items(), key=lambda x: -x[1])
    ranks = rankdata([-s for _, s in items], method='average')
    for i, (m, _) in enumerate(items):
        borda[m] += 4 - ranks[i]
    n_rank += 1

print(f"Borda computed from {n_rank} user rankings")
borda_rank = sorted(borda.items(), key=lambda x: -x[1])

# compare rankings
common = set([m for m, _ in rm_rank]) & set([m for m, _ in borda_rank])
print(f"common models: {len(common)}")

rm_dict = {m: i+1 for i, (m, _) in enumerate(rm_rank)}
borda_dict = {m: i+1 for i, (m, _) in enumerate(borda_rank)}

mods = sorted(common)
rm_r = [rm_dict[m] for m in mods]
borda_r = [borda_dict[m] for m in mods]

tau, p = kendalltau(rm_r, borda_r)

print(f"\ntau = {tau:.3f} (p = {p:.2e})")

if tau > 0.7:
    verdict = "STRONG: DPL prediction supported"
elif tau > 0.5:
    verdict = "MODERATE: partial support"
else:
    verdict = "WEAK: DPL prediction not supported"
print(f"verdict: {verdict}")

print("\n--- creating figure ---")
fig, ax = plt.subplots(figsize=(10, 6))

# plot rankings
df_plot = pd.DataFrame({'model': mods, 'RM': rm_r, 'Borda': borda_r})
df_plot = df_plot.sort_values('RM')

x = range(len(df_plot))
w = 0.35
ax.barh([i - w/2 for i in x], df_plot['RM'], w, label='Reward Model', alpha=0.8)
ax.barh([i + w/2 for i in x], df_plot['Borda'], w, label='Borda Count', alpha=0.8)

ax.set_yticks(x)
ax.set_yticklabels([m[:30] + '...' if len(m) > 30 else m for m in df_plot['model']], fontsize=8)
ax.set_xlabel('Rank (lower = better)')
ax.set_title(f'Reward Model vs Borda Rankings (τ = {tau:.3f})')
ax.legend()
ax.invert_xaxis()

plt.tight_layout()
plt.savefig('figures/fig2_rankings_comparison.png', dpi=150, bbox_inches='tight')
print("saved: figures/fig2_rankings_comparison.png")

# save results
pd.DataFrame({'model': mods, 'rm_rank': rm_r, 'borda_rank': borda_r}).to_csv(
    'data/rankings_comparison.csv', index=False)
pd.DataFrame([{'tau': tau, 'p': p, 'test_acc': acc, 'verdict': verdict}]).to_csv(
    'data/aggregation_results.csv', index=False)
print("saved: data/rankings_comparison.csv, data/aggregation_results.csv")

print("done")
