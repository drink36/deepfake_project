import json
import pandas as pd

with open('/fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata.json', 'r') as f:
    data = json.load(f)
with open('/fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/val_metadata.json', 'r') as f:
    data_val = json.load(f)

df = pd.DataFrame(data)
df_val = pd.DataFrame(data_val)

df_all = pd.concat([df, df_val], ignore_index=True)
df_all = df_all[df_all['modify_type'].isin(['both_modified', 'real', 'visual_modified'])].reset_index(drop=True)

df_real = df_all[df_all['modify_type'] == 'real']
df_modified = df_all[df_all['modify_type'].isin(['both_modified', 'visual_modified'])]

print(f"Total real samples: {len(df_real)}")
print(f"Total modified samples: {len(df_modified)}")

df_real_train = df_real.sample(frac=0.7, random_state=42)
df_real_temp = df_real.drop(df_real_train.index)
df_real_val = df_real_temp.sample(frac=0.5, random_state=42)
df_real_test = df_real_temp.drop(df_real_val.index)
df_modified_train = df_modified.sample(frac=0.7, random_state=42)
df_modified_temp = df_modified.drop(df_modified_train.index)
df_modified_val = df_modified_temp.sample(frac=0.5, random_state=42)
df_modified_test = df_modified_temp.drop(df_modified_val.index)
df_train = pd.concat([df_real_train, df_modified_train], ignore_index=True)
df_val = pd.concat([df_real_val, df_modified_val], ignore_index=True)
df_test = pd.concat([df_real_test, df_modified_test], ignore_index=True)

df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
df_val = df_val.sample(frac=1, random_state=42).reset_index(drop=True)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

train_metadata_filtered = df_train.to_dict(orient='records')
validation_metadata_filtered = df_val.to_dict(orient='records')
test_metadata_filtered = df_test.to_dict(orient='records')

with open('/fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/train_metadata_filtered.json', 'w') as f:
    json.dump(train_metadata_filtered, f, indent=4)
with open('/fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/validation_metadata_filtered.json', 'w') as f:
    json.dump(validation_metadata_filtered, f, indent=4)
with open('/fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus/test_metadata_filtered.json', 'w') as f:
    json.dump(test_metadata_filtered, f, indent=4)

print("Done!")