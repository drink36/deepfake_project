from avdeepfake1m.evaluation import auc
import json
import pandas as pd
if __name__ == "__main__":
    
    videomae_visual = "output/videomae_v2_test_10000_20251215-035913.txt"
    r2plus1d_visual = "output/r2plus1d_test_10000_20251215-040018.txt"
    xception_visual = "output/xception_test_10000_20251215-143319.txt"
    visual_metadata = "result_data/test_metadata_filtered_top10000.json"
    videomae_audio5 = "result_data/videomae_v2_test_10000_20251215-035913_combined.txt"
    r2plus1d_audio5 = "result_data/r2plus1d_test_10000_20251215-040018_combined.txt"
    xception_audio5 = "result_data/xception_test_10000_20251215-143319_combined.txt"
    audio_metadata5 = "result_data/test_metadata_filtered_top10000_combined.json"
    videomae_audio2 = "result_data/videomae_v2_test_10000_20251215-035913_combined_top2000.txt"
    r2plus1d_audio2 = "result_data/r2plus1d_test_10000_20251215-040018_combined_top2000.txt"
    xception_audio2 = "result_data/xception_test_10000_20251215-143319_combined_top2000.txt"
    audio_metadata2 = "result_data/test_metadata_filtered_top10000_combined_top2000.json"
    videomae_testB=0.8103299736976624
    r2plus1d_testB=0.7260900735855103
    xception_testB=0.5787936449050903
    videomae_auc = auc(videomae_visual, visual_metadata, "file","fake_segments")
    r2plus1d_auc = auc(r2plus1d_visual, visual_metadata, "file","fake_segments")
    xception_auc = auc(xception_visual, visual_metadata, "file","fake_segments")
    videomae_5_auc = auc(videomae_audio5, audio_metadata5, "file","fake_segments")
    r2plus1d_5_auc = auc(r2plus1d_audio5, audio_metadata5, "file","fake_segments")
    xception_5_auc = auc(xception_audio5, audio_metadata5, "file","fake_segments")
    videomae_2_auc = auc(videomae_audio2, audio_metadata2, "file","fake_segments")
    r2plus1d_2_auc = auc(r2plus1d_audio2, audio_metadata2, "file","fake_segments")
    xception_2_auc = auc(xception_audio2, audio_metadata2, "file","fake_segments")
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
    
