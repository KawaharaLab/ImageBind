import os

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.datasets import load_digits

# Digitsで試す
digits = load_digits()

# umapで2次元に削減
reducer = umap.UMAP(random_state=42)
reducer.fit(digits.data)
embedding = reducer.transform(digits.data)

# plot
plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24)

# 画像をファイルに保存
out_dir = "data"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "digits_umap.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved UMAP projection to {out_path}")
