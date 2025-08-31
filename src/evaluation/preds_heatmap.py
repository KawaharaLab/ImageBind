import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# run encoder_per_force.py first

plt.rcParams['font.size'] = 12

df = pd.read_csv("/home/user/ImageBind/data/pure/comfy-paper-9/predictions.csv")

y_true = df['true_label']
y_pred = df['pred_label']

labels = [
    "Holds the object.",
    "No contact.",
    "Grabs the object.",
    "Places the object.",
    "Drops the object.",
    "The object slips.",
    "Touches the object.",
]

cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, vmax=100)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

plt.savefig('confusion_matrix_heatmap.png')
