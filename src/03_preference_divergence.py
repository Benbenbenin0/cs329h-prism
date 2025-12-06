# preference divergence: inter-group correlations on model preferences
# expected results: r = 0.934 (age), 0.925 (gender), 0.982 (region)
# outputs: Figure 1, Table 2 stats

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("--- preference divergence ---")

pairs = pd.read_csv('data/pairs.csv')
print(f"pairs: {len(pairs)}, users: {pairs['user_id'].nunique()}")

# minimum appearances threshold for group inclusion
MIN_APPEARANCES = 100

def get_win_rates(df, col, min_apps=MIN_APPEARANCES):
    """Compute model win rates for each demographic group.
    Excludes groups with fewer than min_apps total model appearances.
    """
    # count appearances per group
    group_apps = {}
    for _, r in df.iterrows():
        g = r[col]
        if pd.notna(g):
            group_apps[g] = group_apps.get(g, 0) + 2

    # filter to groups meeting threshold
    valid_groups = [g for g, n in group_apps.items() if n >= min_apps]

    out = []
    for g in valid_groups:
        gdf = df[df[col] == g]
        wins, apps = {}, {}
        for _, r in gdf.iterrows():
            m_a, m_b = r['model_a'], r['model_b']
            apps[m_a] = apps.get(m_a, 0) + 1
            apps[m_b] = apps.get(m_b, 0) + 1
            winner = m_a if r['preference'] == 1 else m_b
            wins[winner] = wins.get(winner, 0) + 1
        for m in apps:
            out.append({
                'demo': col, 'group': g, 'model': m,
                'wins': wins.get(m, 0), 'apps': apps[m],
                'rate': wins.get(m, 0) / apps[m]
            })
    return pd.DataFrame(out)

print("--- computing win rates ---")
wr = pd.concat([get_win_rates(pairs, c) for c in ['age', 'gender', 'region']])

# inter-group correlation
def group_corr(wr_df, col):
    pivot = wr_df[wr_df['demo'] == col].pivot(index='model', columns='group', values='rate')
    return pivot.corr()

print("\n--- inter-group correlations ---")
corrs = []
for demo in ['age', 'gender', 'region']:
    corr = group_corr(wr, demo)
    # get off-diagonal mean
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    vals = corr.where(mask).stack()
    mean_r = vals.mean()
    min_r = vals.min()
    print(f"{demo}: mean r = {mean_r:.3f}, min r = {min_r:.3f}")
    corrs.append({'demographic': demo, 'mean_r': mean_r, 'min_r': min_r})

print("\n--- logistic regression ---")
# exclude "Prefer not to say"
pairs_lr = pairs[pairs['gender'] != 'Prefer not to say'].copy()
model = sm.Logit.from_formula('preference ~ C(age) + C(gender) + C(region)', data=pairs_lr)
result = model.fit(disp=0)

# FDR correction
coefs = result.summary2().tables[1]
pvals = coefs['P>|z|'].values[1:]
names = coefs.index[1:]
reject, padj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

print(f"\nsignificant effects after FDR: {sum(reject)}/{len(pvals)}")

# save logistic results
lr_out = pd.DataFrame({
    'coef': names, 'OR': np.exp(result.params[1:]),
    'p': pvals, 'p_adj': padj, 'sig': reject
})
lr_out.to_csv('data/logistic_regression.csv', index=False)

# bootstrap CIs
# this takes a looooong time (~1 hr on Mac Studio)
print("\n--- bootstrap ---")
np.random.seed(42)
boots = []

for demo in ['age', 'gender', 'region']:
    demo_pairs = pairs[pairs[demo].notna()].copy()
    boot_corrs = []

    for _ in range(1000):
        # resample users
        users = demo_pairs['user_id'].unique()
        boot_users = np.random.choice(users, len(users), replace=True)
        boot_df = pd.concat([demo_pairs[demo_pairs['user_id'] == u] for u in boot_users])

        # compute correlation
        wr_boot = get_win_rates(boot_df, demo)
        corr = group_corr(wr_boot, demo)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        vals = corr.where(mask).stack()
        boot_corrs.append(vals.mean())

    ci_lo, ci_hi = np.percentile(boot_corrs, [2.5, 97.5])
    point = corrs[[r['demographic'] for r in corrs].index(demo)]['mean_r']
    print(f"{demo}: r = {point:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    boots.append({'demographic': demo, 'r': point, 'ci_lo': ci_lo, 'ci_hi': ci_hi})

pd.DataFrame(boots).to_csv('data/correlation_bootstrap.csv', index=False)

# figure 1
print("\n--- creating figure ---")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, demo in enumerate(['age', 'gender', 'region']):
    ax = axes[i]
    pivot = wr[wr['demo'] == demo].pivot(index='model', columns='group', values='rate')
    # sort by mean win rate
    pivot['mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean', ascending=False).drop('mean', axis=1).head(12)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5, ax=ax, cbar=False)

    r = [x['mean_r'] for x in corrs if x['demographic'] == demo][0]
    ax.set_title(f'{demo.title()} (mean r = {r:.3f})', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Model' if i == 0 else '')

plt.tight_layout()
plt.savefig('figures/fig1_preference_divergence.png', dpi=150, bbox_inches='tight')
print("saved: figures/fig1_preference_divergence.png")

# save win rates
wr.to_csv('data/model_win_rates.csv', index=False)
print("saved: data/model_win_rates.csv")

print("done")
