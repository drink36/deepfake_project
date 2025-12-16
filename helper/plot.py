# plot_grouped_auc.py
import numpy as np
import matplotlib.pyplot as plt

models = ["VideoMAE", "R2Plus1D", "Xception"]

visual = [0.9989, 0.9169, 0.9161]
audio_2k = [0.9110, 0.8802, 0.8118]
audio_5k = [0.8377, 0.8493, 0.7273]
testb = [0.8103, 0.7261, 0.5788]

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
