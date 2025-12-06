# embedding validation: test if embeddings capture preference-relevant features
# expected result: r = -0.285 (dissimilar responses -> stronger preferences)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

print("--- embedding validation ---")

pairs = pd.read_csv('data/pairs.csv')
print(f"pairs: {len(pairs)}")

# get unique responses
resp_a = pairs[['utt_id_a', 'response_a']].rename(columns={'utt_id_a': 'id', 'response_a': 'text'})
resp_b = pairs[['utt_id_b', 'response_b']].rename(columns={'utt_id_b': 'id', 'response_b': 'text'})
resps = pd.concat([resp_a, resp_b]).drop_duplicates('id')
print(f"unique responses: {len(resps)}")

print("--- encoding ---")
model = SentenceTransformer('all-mpnet-base-v2')
embs = model.encode(resps['text'].tolist(), show_progress_bar=True, batch_size=64)
emb_lookup = dict(zip(resps['id'], embs))
print(f"embedding dim: {embs.shape[1]}")

def get_sim(row):
    ea, eb = emb_lookup.get(row['utt_id_a']), emb_lookup.get(row['utt_id_b'])
    if ea is None or eb is None:
        return np.nan
    return cosine_similarity([ea], [eb])[0, 0]

pairs['cos_sim'] = pairs.apply(get_sim, axis=1)

# preference strength = normalized margin
pairs['pref_str'] = pairs['margin'] / 99.0

# correlation analysis
valid = pairs['cos_sim'].notna()
sim = pairs.loc[valid, 'cos_sim']
strength = pairs.loc[valid, 'pref_str']

r, p = stats.pearsonr(sim, strength)
rho, p_rho = stats.spearmanr(sim, strength)

print(f"\ncos sim: mean={sim.mean():.4f}, std={sim.std():.4f}")
print(f"pearson r = {r:.3f} (p = {p:.2e})")
print(f"spearman rho = {rho:.3f} (p = {p_rho:.2e})")

if r < -0.1 and p < 0.05:
    print("negative correlation as expected")
else:
    print("unexpected result")

# save
out = pairs[['conversation_id', 'user_id', 'cos_sim', 'pref_str', 'margin', 'preference']]
out.to_csv('data/embedding_validation.csv', index=False)
print("\nsaved: data/embedding_validation.csv")

print("done")
