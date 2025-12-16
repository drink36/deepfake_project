import json
import sys
import os

def create_top_n_json(input_json, n, output_json=None): 
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    top_n = data[:n]
    filename = os.path.basename(input_json)
    if output_json is None:
        base, ext = os.path.splitext(filename)
        output_json = f"{base}_top{n}{ext}"
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(top_n, f, indent=2)



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python cut_json.py <input.json> <n>")
        sys.exit(1)

    input_json = sys.argv[1]
    n = int(sys.argv[2])

    create_top_n_json(input_json, n)
