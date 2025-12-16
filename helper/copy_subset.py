import os
import json
import shutil
from tqdm import tqdm

def copy_files(json_file, source_root, dest_root):
    print(f"📄 Reading file list from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"📂 Source Root: {source_root}")
    print(f"📦 Dest Root:   {dest_root}")
    print(f"🚀 Total files to copy: {len(data)}")

    
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)

    success_count = 0
    missing_count = 0
    
    for item in tqdm(data):
        file_rel_path = item['file']
        split_folder = item['split']
        
        
        
        src_path = os.path.join(source_root, split_folder, file_rel_path)
        
        if not os.path.exists(src_path):
        
            src_path_alt = os.path.join(source_root, file_rel_path)
            if os.path.exists(src_path_alt):
                src_path = src_path_alt
        
        
        dest_path = os.path.join(dest_root, split_folder, file_rel_path)
        
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dest_path) 
                success_count += 1
            except Exception as e:
                print(f"❌ Error copying {src_path}: {e}")
        else:
            print(f"⚠️  Missing source file: {src_path}")
            missing_count += 1

    print("\n" + "="*30)
    print(f"✅ Copy Completed!")
    print(f"   Success: {success_count}")
    print(f"   Missing: {missing_count}")
    print(f"📂 Output folder: {dest_root}")
    print("="*30)

if __name__ == "__main__":
    
    MY_JSON_FILE = 'train_subset.json' 
    MY_JSON_FILE2 = 'test_subset.json'
    MY_JSON_FILE3 = 'val_subset.json'
    SOURCE_DATA_ROOT = '/fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus'
    
    DEST_DIR_NAME = 'test_dataset'

    copy_files(MY_JSON_FILE, SOURCE_DATA_ROOT, DEST_DIR_NAME)
    copy_files(MY_JSON_FILE2, SOURCE_DATA_ROOT, DEST_DIR_NAME)
    copy_files(MY_JSON_FILE3, SOURCE_DATA_ROOT, DEST_DIR_NAME)