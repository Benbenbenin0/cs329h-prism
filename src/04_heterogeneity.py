# heterogeneity: variance ratio test and opponent confound analysis
# expected results: variance ratio ~2 (1.74-2.29), opponent confound 7.9%

import pandas as pd
import numpy as np
from collections import defaultdict

print("--- heterogeneity ---")

pairs = pd.read_csv('data/pairs.csv')
print(f"pairs: {len(pairs)}")

# filter users with enough data and variation
user_stats = pairs.groupby('user_id').agg({
    'preference': ['count', 'sum', 'mean']
}).reset_index()
user_stats.columns = ['user_id', 'n', 'wins', 'rate']
user_stats['has_var'] = (user_stats['wins'] > 0) & (user_stats['wins'] < user_stats['n'])
valid = user_stats[(user_stats['n'] >= 5) & user_stats['has_var']]['user_id']
df = pairs[pairs['user_id'].isin(valid)].copy()
print(f"valid users: {len(valid)}, pairs: {len(df)}")

# build user-model win rates
user_model = defaultdict(lambda: defaultdict(lambda: {'w': 0, 't': 0}))
for _, r in df.iterrows():
    uid = r['user_id']
    if r['preference'] == 1:
        user_model[uid][r['model_a']]['w'] += 1
        user_model[uid][r['model_a']]['t'] += 1
        user_model[uid][r['model_b']]['t'] += 1
    else:
        user_model[uid][r['model_b']]['w'] += 1
        user_model[uid][r['model_b']]['t'] += 1
        user_model[uid][r['model_a']]['t'] += 1

# users with 10+ models rated
users_10plus = [u for u in user_model if len(user_model[u]) >= 10]
print(f"users with 10+ models: {len(users_10plus)}")

print("\n--- variance ratios ---")
models = df['model_a'].unique()
results = []

# TODO: could try this with vectorization
for m in models:
    rates, ns = [], []
    for u in users_10plus:
        if m in user_model[u] and user_model[u][m]['t'] >= 3:
            rate = user_model[u][m]['w'] / user_model[u][m]['t']
            rates.append(rate)
            ns.append(user_model[u][m]['t'])

    if len(rates) >= 30:
        rates = np.array(rates)
        ns = np.array(ns)
        obs_var = np.var(rates)
        p_mean = np.mean(rates)
        exp_var = np.mean([p_mean * (1 - p_mean) / n for n in ns])
        ratio = obs_var / exp_var if exp_var > 0 else 0

        # bootstrap CI
        np.random.seed(42)
        boot_ratios = []
        for _ in range(1000):
            idx = np.random.choice(len(rates), len(rates), replace=True)
            br, bn = rates[idx], ns[idx]
            bov = np.var(br)
            bp = np.mean(br)
            bev = np.mean([bp * (1 - bp) / n for n in bn])
            if bev > 0:
                boot_ratios.append(bov / bev)
        ci_lo, ci_hi = np.percentile(boot_ratios, [2.5, 97.5])

        results.append({
            'model': m, 'n_users': len(rates), 'ratio': ratio,
            'ci_lo': ci_lo, 'ci_hi': ci_hi, 'sig': ci_lo > 1.0
        })

results_df = pd.DataFrame(results).sort_values('ratio', ascending=False)

print(f"\n{'model':<40} {'ratio':>8} {'95% CI':>20} {'sig':>5}")
print("-" * 75)
for _, r in results_df.head(10).iterrows():
    sig = '*' if r['sig'] else ''
    ci = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
    print(f"{r['model'][:40]:<40} {r['ratio']:>8.2f} {ci:>20} {sig:>5}")

print(f"\nvariance ratio range: {results_df['ratio'].min():.2f} - {results_df['ratio'].max():.2f}")
print(f"mean ratio: {results_df['ratio'].mean():.2f}")
print(f"models with significant heterogeneity: {results_df['sig'].sum()}/{len(results_df)}")

results_df.to_csv('data/variance_ratios.csv', index=False)
print("\nsaved: data/variance_ratios.csv")

print("\n--- confound ---")
conf_res = []

for m in models:
    # global win rate against each opponent
    opp_win_rates = defaultdict(lambda: {'w': 0, 't': 0})
    for _, r in df.iterrows():
        if r['model_a'] == m:
            opp = r['model_b']
            opp_win_rates[opp]['t'] += 1
            if r['preference'] == 1:
                opp_win_rates[opp]['w'] += 1
        elif r['model_b'] == m:
            opp = r['model_a']
            opp_win_rates[opp]['t'] += 1
            if r['preference'] == 0:
                opp_win_rates[opp]['w'] += 1

    # user-level data
    user_data = defaultdict(lambda: {'w': 0, 't': 0, 'opps': []})
    for _, r in df.iterrows():
        if r['model_a'] == m:
            uid = r['user_id']
            user_data[uid]['t'] += 1
            user_data[uid]['opps'].append(r['model_b'])
            if r['preference'] == 1:
                user_data[uid]['w'] += 1
        elif r['model_b'] == m:
            uid = r['user_id']
            user_data[uid]['t'] += 1
            user_data[uid]['opps'].append(r['model_a'])
            if r['preference'] == 0:
                user_data[uid]['w'] += 1

    # filter
    ok_users = [u for u in user_data if user_data[u]['t'] >= 3]

    if len(ok_users) >= 30:
        observed = []
        expected = []
        for u in ok_users:
            d = user_data[u]
            observed.append(d['w'] / d['t'])
            exp = 0 # expected based on opponent mix
            for opp in d['opps']:
                if opp_win_rates[opp]['t'] > 0:
                    exp += opp_win_rates[opp]['w'] / opp_win_rates[opp]['t']
            expected.append(exp / len(d['opps']))

        obs = np.array(observed)
        exp = np.array(expected)
        obs_var = np.var(obs)
        exp_var = np.var(exp)
        resid_var = np.var(obs - exp)

        if obs_var > 0:
            conf_res.append({
                'model': m, 'n_users': len(ok_users),
                'pct_explained': 100 * exp_var / obs_var,
                'pct_residual': 100 * resid_var / obs_var
            })

confound_df = pd.DataFrame(conf_res)
mean_explained = confound_df['pct_explained'].mean()
mean_residual = confound_df['pct_residual'].mean()

print(f"\nmean % explained by opponent mix: {mean_explained:.1f}%")
print(f"mean % residual (genuine): {mean_residual:.1f}%")

confound_df.to_csv('data/opponent_confound.csv', index=False)
print("saved: data/opponent_confound.csv")

print(f"\nvariance ratio: {results_df['ratio'].min():.2f} - {results_df['ratio'].max():.2f}")
print(f"confound: {mean_explained:.1f}% explained, {mean_residual:.1f}% genuine")

print("done")
