import torch

in_file  = "output/videomae_v2_testB_46293_20251213-165228.txt"
out_file = "output/videomae_v2_testB_46293_20251213-165228_prob.txt"

names = []
logits = []

with open(in_file, "r") as f:
    for line in f:
        name, val = line.strip().split(";")
        names.append(name)
        logits.append(float(val))

logits = torch.tensor(logits, dtype=torch.float32)
probs = torch.sigmoid(logits)

with open(out_file, "w") as f:
    for name, p in zip(names, probs):
        f.write(f"{name};{p.item()}\n")
