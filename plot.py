# plot_grouped_auc.py
import numpy as np
import matplotlib.pyplot as plt

models = ["VideoMAE", "R2Plus1D", "Xception"]

visual = [0.9985, 0.9230, 0.7829]
audio_2k = [0.9115, 0.8871, 0.7054]
audio_5k = [0.8388, 0.8569, 0.6568]
testb = [0.8103, 0.7261, 0.5729]

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - 1.5*width, visual, width, label="Visual")
plt.bar(x - 0.5*width, audio_2k, width, label="Visual + Audio 2K")
plt.bar(x + 0.5*width, audio_5k, width, label="Visual + Audio 5K")
plt.bar(x + 1.5*width, testb, width, label="TestB (Visual)")

plt.xticks(x, models)
plt.ylabel("AUC")
plt.ylim(0.5, 1.0)
plt.title("Model Performance Comparison (AUC)")
plt.legend()
plt.tight_layout()

plt.savefig("grouped_model_auc.png", dpi=300)
plt.show()
