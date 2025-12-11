import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from data.dataset import VideoMetadata

with open("/fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json") as f:
    data = json.load(f)

metas = [VideoMetadata(**x) for x in data]

fake = sum(1 for m in metas if len(m.fake_periods) > 0)
real = len(metas) - fake
print("train videos:", len(metas), "video-fake:", fake, "real:", real)

for m in metas[:20]:
    print(m.file, "fake_periods:", m.fake_periods)