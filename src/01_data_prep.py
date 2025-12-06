# data prep: load PRISM, create pairwise preferences with demographics
# outputs: data/pairs.csv, data/turn0_pairs.csv

import pandas as pd
import numpy as np
from datasets import load_dataset
from itertools import combinations

print("--- data prep ---")
print("--- loading prism ---")
utt = load_dataset("HannahRoseKirk/prism-alignment", "utterances")['train'].to_pandas()
survey = load_dataset("HannahRoseKirk/prism-alignment", "survey")['train'].to_pandas()
print(f"utterances: {len(utt)}, survey: {len(survey)}")

print("--- creating pairs ---")
turn0 = utt[utt['turn'] == 0].copy()

pairs = []
for cid, grp in turn0.groupby('conversation_id'):
    rows = grp[['utterance_id', 'model_name', 'score', 'user_id', 'model_response', 'user_prompt']].values.tolist()
    for (id_a, m_a, s_a, uid, r_a, prompt), (id_b, m_b, s_b, _, r_b, _) in combinations(rows, 2):
        if pd.notna(s_a) and pd.notna(s_b):
            margin = abs(s_a - s_b)
            pref = 'A' if s_a > s_b else ('B' if s_b > s_a else 'tie')
            pairs.append({
                'conversation_id': cid,
                'user_id': uid,
                'utt_id_a': id_a,
                'utt_id_b': id_b,
                'model_a': m_a,
                'model_b': m_b,
                'score_a': s_a,
                'score_b': s_b,
                'margin': margin,
                'preferred': pref,
                'response_a': r_a,
                'response_b': r_b,
                'prompt': prompt
            })

df = pd.DataFrame(pairs)
print(f"pairs created: {len(df)}")

# save turn0 pairs for borda analysis
df.to_csv('data/turn0_pairs.csv', index=False)
print("saved: data/turn0_pairs.csv")

print("--- merging demographics ---")
demo_cols = ['user_id', 'age', 'gender', 'education', 'study_locale', 'location']
demo_df = survey[demo_cols].copy()
df = df.merge(demo_df, on='user_id', how='left')

def get_region(row):
    """Assign region based on study_locale and location fields."""
    locale = str(row.get('study_locale', '')).lower()
    location = str(row.get('location', '')).lower()

    if 'us' in locale or 'united states' in location:
        return 'US'
    elif 'uk' in locale or 'united kingdom' in location or 'england' in location:
        return 'UK'
    else:
        return 'Other'

df['region'] = df.apply(get_region, axis=1)

# filter to T_mid threshold (margin > 27, strict inequality)
T_MID = 27
df = df[df['margin'] > T_MID].copy()
df['preference'] = (df['score_a'] > df['score_b']).astype(int)  # 1=A, 0=B

print(f"pairs after T_mid filter (margin > {T_MID}): {len(df)}")
print(f"unique users: {df['user_id'].nunique()}")

# save main dataset
df.to_csv('data/pairs.csv', index=False)
print("saved: data/pairs.csv")

print("\n--- demographics ---")
for col in ['age', 'gender', 'region']:
    print(f"\n{col}:")
    vc = df.groupby(col)['user_id'].nunique()
    for k, v in vc.items():
        print(f"  {k}: {v} users")

# save summary
demo_summary = []
for col in ['age', 'gender', 'region']:
    for grp, cnt in df.groupby(col)['user_id'].nunique().items():
        demo_summary.append({'category': col, 'group': grp, 'n_users': cnt})
pd.DataFrame(demo_summary).to_csv('data/demographic_summary.csv', index=False)
print("\nsaved: data/demographic_summary.csv")

print("\ndone")
