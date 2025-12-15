from avdeepfake1m.evaluation import auc
import json
import pandas as pd
if __name__ == "__main__":
    
    videomae_visual = "output/videomae_v2_val_10000_20251214-011831.txt"
    r2plus1d_visual = "output/r2plus1d_test_10000_20251214-144857.txt"
    visual_metadata = "validation_metadata_filtered_top10000.json"
    videomae_audio5 = "videomae_v2_val_10000_20251214-011831_combined.txt"
    r2plus1d_audio5 = "r2plus1d_test_10000_20251214-144857_combined.txt"
    audio_metadata5 = "validation_metadata_filtered_top10000_combined.json"
    videomae_audio2 = "videomae_v2_val_10000_20251214-011831_combined_top2000.txt"
    r2plus1d_audio2 = "r2plus1d_test_10000_20251214-144857_combined_top2000.txt"
    audio_metadata2 = "validation_metadata_filtered_top10000_combined_top2000.json"
    videomae_testB=0.8103299736976624
    r2plus1d_testB=0.7260900735855103
    xception_testB=0.5729
    videomae_auc = auc(videomae_visual, visual_metadata, "file","fake_segments")
    r2plus1d_auc = auc(r2plus1d_visual, visual_metadata, "file","fake_segments")
    xception_auc = 0.7829
    videomae_5_auc = auc(videomae_audio5, audio_metadata5, "file","fake_segments")
    r2plus1d_5_auc = auc(r2plus1d_audio5, audio_metadata5, "file","fake_segments")
    xception_5_auc = 0.6568
    videomae_2_auc = auc(videomae_audio2, audio_metadata2, "file","fake_segments")
    r2plus1d_2_auc = auc(r2plus1d_audio2, audio_metadata2, "file","fake_segments")
    xception_2_auc = 0.7054
    print(f"VideoMAE AUC: {videomae_auc:.4f}")
    print(f"R2Plus1D AUC: {r2plus1d_auc:.4f}")
    print(f"Xception AUC: {xception_auc:.4f}")
    print(f"VideoMAE + Audio5K AUC: {videomae_5_auc:.4f}")
    print(f"R2Plus1D + Audio5K AUC: {r2plus1d_5_auc:.4f}")
    print(f"Xception + Audio5K AUC: {xception_5_auc:.4f}")
    print(f"VideoMAE + Audio2K AUC: {videomae_2_auc:.4f}")
    print(f"R2Plus1D + Audio2K AUC: {r2plus1d_2_auc:.4f}")
    print(f"Xception + Audio2K AUC: {xception_2_auc:.4f}")
    print(f"TestB VideoMAE AUC: {videomae_testB:.4f}")
    print(f"TestB R2Plus1D AUC: {r2plus1d_testB:.4f}")
    print(f"TestB Xception AUC: {xception_testB:.4f}")
    # 將結果存成 txt 檔
    results = {
        "VideoMAE AUC": videomae_auc,
        "R2Plus1D AUC": r2plus1d_auc,
        "Xception AUC": xception_auc,
        "VideoMAE + Audio5K AUC": videomae_5_auc,
        "R2Plus1D + Audio5K AUC": r2plus1d_5_auc,
        "Xception + Audio5K AUC": xception_5_auc,
        "VideoMAE + Audio2K AUC": videomae_2_auc,
        "R2Plus1D + Audio2K AUC": r2plus1d_2_auc,
        "Xception + Audio2K AUC": xception_2_auc,
        "TestB VideoMAE AUC": videomae_testB,
        "TestB R2Plus1D AUC": r2plus1d_testB,
        "TestB Xception AUC": xception_testB
    }
    with open("evaluation_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
    print("Evaluation results saved to evaluation_results.txt")
    
