import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# === 1. Load Data ===
path_gen = "C2GAM_training/data/D_gen.csv"
path_obs = "C2GAM_training/data/D_obs.csv"
path_rep = "C2GAM_training/data/D_rep.csv"

gen = pd.read_csv(path_gen)
obs = pd.read_csv(path_obs)
rep = pd.read_csv(path_rep)

# === 2. Select covariates ===
X_cols = [f"X2_{i}" for i in range(1, 7)]  # year, region, age, sex, married, edu

def prepare(df, label):
    df = df.copy()
    df["source"] = label
    return df[["X1", "Y"] + X_cols + ["source"]]

# Dobs는 Drep + Dgen 개수만큼만 사용
obs_sample_size = min(len(rep) + len(gen), len(obs))
obs_sampled = obs.sample(n=obs_sample_size, random_state=42)

df_all = pd.concat([
    prepare(rep, "Drep"),
    prepare(gen, "Dgen"),
    prepare(obs_sampled, "Dobs"),
], axis=0).reset_index(drop=True)

# === 3. Dimensionality Reduction (t-SNE 1D for X summary) ===
tsne = TSNE(n_components=1, perplexity=30, random_state=42)
x_reduced = tsne.fit_transform(df_all[X_cols].fillna(0))
df_all["X_reduced"] = x_reduced[:, 0]

# === 4. 3D Plot (T, X_reduced, Y) ===
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

colors = {
    "Drep": ("blue", "The Representative Data"),
    "Dgen": ("orange", "The Generated Samples"),
    "Dobs": ("green", "The Observational Data")
}

for source, (color, label) in colors.items():
    subset = df_all[df_all["source"] == source]
    ax.scatter(subset["X1"], subset["X_reduced"], subset["Y"],
               c=color, label=label, alpha=0.7, s=30)

ax.set_xlabel("T")
ax.set_ylabel("X (Reduced)")
ax.set_zlabel("Y")
ax.set_title("Joint Distribution of T, X, Y across Dobs, Drep, Dgen")
ax.legend()
plt.tight_layout()
plt.show()
plt.savefig("C2GAM_project/results/plot_distribution.png", dpi=300)