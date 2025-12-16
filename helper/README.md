# Helper Scripts

This directory contains utility scripts for data processing, file management, and results analysis.

## 📄 Scripts

### `combine.py`
Merges two result files (either JSON or TXT).
*   **Usage:** `python helper/combine.py <input1> <input2> [optional_n]`
*   **Function:** Concatenates the content of two files. If `n` is specified, it takes only the top `n` items from the second file. Useful for combining predictions from different subsets (e.g., visual-only + audio-augmented).

### `compare.py`
Compares the performance (AUC) of different models and scenarios using hardcoded paths.
*   **Usage:** `python helper/compare.py` (Edit the file paths in the script before running).
*   **Function:** Loads prediction files for VideoMAE, R(2+1)D, and Xception across various test sets (Visual, Audio-Visual 2K/5K, TestB). Calculates AUC scores for each and saves the summary to `evaluation_results.txt`.

### `copy_subset.py`
Copies a subset of the dataset from a source directory to a destination based on a JSON list.
*   **Usage:** `python helper/copy_subset.py` (Edit the `if __name__ == "__main__":` block to set paths).
*   **Function:** Reads a JSON file containing video metadata, locates the files in the source directory, and copies them to a new folder structure. Useful for creating smaller datasets for local testing.

### `cut_json.py`
Creates a smaller version of a metadata JSON file containing the top `N` entries.
*   **Usage:** `python helper/cut_json.py <input.json> <n>`
*   **Function:** Reads a JSON file, slices the first `n` items, and saves them to `filename_topN.json`.

### `data_preprocessor.py`
Splits the raw dataset into Train, Validation, and Test sets based on modification types.
*   **Usage:** `python helper/data_preprocessor.py` (Edit the `if __name__ == "__main__":` block to configure paths and filters).
*   **Function:**
    1.  Loads raw `train_metadata.json` and `val_metadata.json`.
    2.  Filters videos by modification type (e.g., 'real', 'visual_modified', 'audio_modified').
    3.  Splits data into Train/Val/Test (default 70/15/15) while maintaining balance between real and fake samples.
    4.  Saves the new split configurations to JSON files.

### `plot.py`
Visualizes the AUC comparison results.
*   **Usage:** `python helper/plot.py` (Edit the hardcoded AUC scores in the script before running).
*   **Function:** Generates a bar chart (`grouped_model_auc.png`) comparing the AUC scores of the three models across different evaluation scenarios.

