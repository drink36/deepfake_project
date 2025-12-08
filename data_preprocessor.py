import json
import pandas as pd
import os
from typing import List, Dict, Optional

class DataPreprocessor:
    def __init__(self, 
                 root_dir: str, 
                 train_json_name: str = 'train_metadata.json', 
                 val_json_name: str = 'val_metadata.json'):
        """
        初始化預處理器，設定檔案路徑。
        """
        self.root_dir = root_dir
        self.train_path = os.path.join(root_dir, train_json_name)
        self.val_path = os.path.join(root_dir, val_json_name)
        
        self.splits: Dict[str, List[Dict]] = {} 

    def load_and_process_splits(self, 
                                valid_modify_types: List[str] = ['both_modified', 'real', 'visual_modified'],
                                ratios: tuple = (0.7, 0.15, 0.15),
                                seed: int = 42):

        print(f"Loading raw data from {self.root_dir}...")
        

        try:
            with open(self.train_path, 'r') as f: data_train = json.load(f)
            with open(self.val_path, 'r') as f: data_val = json.load(f)
        except FileNotFoundError as e:
            print(f"❌ Error: no file - {e}")
            return


        df_all = pd.concat([pd.DataFrame(data_train), pd.DataFrame(data_val)], ignore_index=True)
        
        print(f"Filtering modify_types: {valid_modify_types}...")
        df_all = df_all[df_all['modify_type'].isin(valid_modify_types)].reset_index(drop=True)

        df_real = df_all[df_all['modify_type'] == 'real']
        df_modified = df_all[df_all['modify_type'].isin(['both_modified', 'visual_modified'])]

        print(f"-> Real samples: {len(df_real)}")
        print(f"-> Modified samples: {len(df_modified)}")

        train_ratio, val_ratio, test_ratio = ratios
        
        def split_df(df):

            train = df.sample(frac=train_ratio, random_state=seed)
            temp = df.drop(train.index)

            val_frac = val_ratio / (val_ratio + test_ratio)
            val = temp.sample(frac=val_frac, random_state=seed)

            test = temp.drop(val.index)
            return train, val, test

        real_splits = split_df(df_real)
        mod_splits = split_df(df_modified)

        self.splits['train'] = pd.concat([real_splits[0], mod_splits[0]]).sample(frac=1, random_state=seed).to_dict(orient='records')
        self.splits['val']   = pd.concat([real_splits[1], mod_splits[1]]).sample(frac=1, random_state=seed).to_dict(orient='records')
        self.splits['test']  = pd.concat([real_splits[2], mod_splits[2]]).sample(frac=1, random_state=seed).to_dict(orient='records')

        print(f"✅ Splitting Done! Sizes: Train={len(self.splits['train'])}, Val={len(self.splits['val'])}, Test={len(self.splits['test'])}")

    def save_main_splits(self):
        if not self.splits:
            print("⚠️ Data not loaded. Run load_and_process_splits() first.")
            return

        for name, data in self.splits.items():
            fname = f"{name}_metadata_filtered.json" 
            if name == 'val': fname = 'validation_metadata_filtered.json'
            
            out_path = os.path.join(self.root_dir, fname)
            with open(out_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved split JSON: {fname}")

    def generate_eval_set(self, 
                          source_split: str = 'val', 
                          filter_origin_split: str = 'train', 
                          take_num: int = 300,
                          output_json_name: str = 'eval_subset.json',
                          output_txt_name: str = 'eval_list.txt'):
        if source_split not in self.splits:
            print(f"❌ Error: Split '{source_split}' not found.")
            return

        print(f"\nGenerating Evaluation Set (Source: {source_split}, Filter: {filter_origin_split}, Count: {take_num})...")

        source_data = self.splits[source_split]
        
        filtered_data = [item for item in source_data if item.get('split') == filter_origin_split]
        
        final_dataset = filtered_data[:take_num]
        
        actual_count = len(final_dataset)
        print(f"-> Selected {actual_count} samples.")

        json_path = os.path.join(self.root_dir, output_json_name) 
        with open(output_json_name, 'w') as f:
            json.dump(final_dataset, f, indent=2)
        print(f"✅ Saved JSON: {output_json_name}")

        with open(output_txt_name, 'w') as f:
            for item in final_dataset:
                f.write(item['file'] + '\n')
        print(f"✅ Saved TXT : {output_txt_name}")
        

if __name__ == "__main__":
    
    MY_ROOT_DIR = '/fs/scratch/PAS3162/drink36/AV-Deepfake1M-PlusPlus'
    
    
    processor = DataPreprocessor(root_dir=MY_ROOT_DIR)

    
    processor.load_and_process_splits()
    
    
    processor.save_main_splits()

    processor.generate_eval_set(
        source_split='train',         
        filter_origin_split='train',
        take_num=5000,
        output_json_name='train_subset.json',
        output_txt_name='train_subset.txt'
    )
    processor.generate_eval_set(
        source_split='val',         
        filter_origin_split='train',
        take_num=1000,
        output_json_name='val_subset.json',
        output_txt_name='val_subset.txt'
    )
    processor.generate_eval_set(
        source_split='test',         
        filter_origin_split='train',
        take_num=1000,
        output_json_name='test_subset.json',
        output_txt_name='test_subset.txt'
    )