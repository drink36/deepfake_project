import json
import sys
import os

def combine_json(input_json, input2_json, output_json=None, n=None):
    # 讀取原始 JSON（必須是 list 格式）
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(input2_json, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    # 合併兩個列表
    if n is not None:
        combined = data+ data2[:n]
    else:
        combined = data + data2
    filename = os.path.basename(input_json)

    # 自動產生輸出檔名
    if output_json is None:
        base, ext = os.path.splitext(filename)
        if n is not None:
            output_json = f"{base}_combined_top{n}{ext}"
        else:
            output_json = f"{base}_combined{ext}"

    # 寫出
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"✔ 已建立：{output_json} （共 {len(combined)} 行）")

def combine_txt(input_txt, input2_txt, output_txt=None, n=None):
    # 讀取原始 TXT
    with open(input_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(input2_txt, "r", encoding="utf-8") as f:
        lines2 = f.readlines()
    # 合併兩個列表
    if n is not None:
        combined = lines + lines2[:n]
    else:
        combined = lines + lines2
    filename = os.path.basename(input_txt)

    # 自動產生輸出檔名
    if output_txt is None:
        base, ext = os.path.splitext(filename)
        if n is not None:
            output_txt = f"{base}_combined_top{n}{ext}"
        else:
            output_txt = f"{base}_combined{ext}"

    # 寫出
    with open(output_txt, "w", encoding="utf-8") as f:
        f.writelines(combined)

    print(f"✔ 已建立：{output_txt} （共 {len(combined)} 行）")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python combine.py <input.json> <input2.json> or <input.txt> <input2.txt> with optional n")
        sys.exit(1)

    input = sys.argv[1]
    input2 = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else None
    if input.endswith(".txt") and input2.endswith(".txt"):
        combine_txt(input, input2, n=n)
    elif input.endswith(".json") and input2.endswith(".json"):
        combine_json(input, input2, n=n)
