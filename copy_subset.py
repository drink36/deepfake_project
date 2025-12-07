import os
import json
import shutil
from tqdm import tqdm

def copy_files(json_file, source_root, dest_root):
    """
    根據 JSON 清單，從 source_root 複製檔案到 dest_root，並保持目錄結構。
    """
    print(f"📄 Reading file list from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"📂 Source Root: {source_root}")
    print(f"📦 Dest Root:   {dest_root}")
    print(f"🚀 Total files to copy: {len(data)}")

    # 確保目標根目錄存在
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)

    success_count = 0
    missing_count = 0
    
    for item in tqdm(data):
        file_rel_path = item['file'] # 例如 "train/001.mp4"
        split_folder = item['split'] # 例如 "train" (這通常包含在 file 路徑裡，或者需要拼湊)
        
        # 1. 拼湊原始完整路徑
        # 注意：根據你的 dataset.py 邏輯，路徑通常是 data_root/split/filename
        # 但如果 JSON 的 'file' 已經包含了 'train/xxx.mp4'，那就要小心不要重複拼湊
        # 這裡假設你的 JSON file 欄位只有檔名 (e.g., "video_123.mp4")
        # 或是包含了子資料夾 (e.g., "df/video_123.mp4")
        
        # 為了保險，我們依賴 split 欄位來找原始檔案
        src_path = os.path.join(source_root, split_folder, file_rel_path)
        
        # 如果檔案不在 split 資料夾下，可能 JSON 的 file 欄位已經包含了路徑？
        # 你可以先 print 出來檢查，或者寫個簡單的 fallback
        if not os.path.exists(src_path):
            # 嘗試直接用 file 欄位拼湊 (有些 dataset 結構不同)
            src_path_alt = os.path.join(source_root, file_rel_path)
            if os.path.exists(src_path_alt):
                src_path = src_path_alt
        
        # 2. 設定目標路徑
        # 我們希望在 mini_dataset 裡也保持一樣的結構 (例如 mini_dataset/train/video.mp4)
        dest_path = os.path.join(dest_root, split_folder, file_rel_path)
        
        # 確保目標檔案的父目錄存在
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # 3. 執行複製
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dest_path) # copy2 會保留檔案時間資訊
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
    print("現在你可以把這個資料夾壓縮並下載了！")

if __name__ == "__main__":
    # ================= 設定區 =================
    # 1. 你的 JSON 檔 (裡面只有那 300 筆)
    MY_JSON_FILE = 'subset.json' 
    
    # 2. 原始 1.4TB 數據的根目錄 (Server 上的位置)
    # 根據之前的對話，應該是這個：
    SOURCE_DATA_ROOT = '/fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus'
    
    # 3. 你想要複製到哪裡 (當前目錄下的一個新資料夾)
    DEST_DIR_NAME = 'mini_dataset'
    # =========================================

    copy_files(MY_JSON_FILE, SOURCE_DATA_ROOT, DEST_DIR_NAME)