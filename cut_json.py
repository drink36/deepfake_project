import json
import sys
import os

def create_top_n_json(input_json, n, output_json=None):
    # 讀取原始 JSON（必須是 list 格式）
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 取前 n 筆
    print(f"🔍 原始資料共有 {len(data)} 筆")
    top_n = data[:n]
    filename = os.path.basename(input_json)  # 拿掉路徑，只留檔名

    # 自動產生輸出檔名
    if output_json is None:
        base, ext = os.path.splitext(filename)
        output_json = f"{base}_top{n}{ext}"

    # 寫出
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(top_n, f, indent=2)

    print(f"✔ 已建立：{output_json} （共 {len(top_n)} 行）")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python cut_json.py <input.json> <n>")
        sys.exit(1)

    input_json = sys.argv[1]
    n = int(sys.argv[2])

    create_top_n_json(input_json, n)
