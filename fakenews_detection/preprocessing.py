import os
import re
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data resources quietly
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed, proceeding without lemmatization fallback. Error: {e}")

def clean_text(text):
    """
    Cleans the input text by removing HTML tags, URLs, and extra whitespaces.
    Preserves multilingual characters (like Vietnamese accents) and common punctuation.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Clean up multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Apply English lemmatizer if wordnet is available
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        text = " ".join([lemmatizer.lemmatize(word) for word in words])
    except Exception:
        pass  # Fallback if nltk lemmatizer fails
        
    return text

def load_and_combine_data(dataset_dir):
    """
    Loads gossipcop and politifact CSV files and combines them.
    Assigns: Real = 0, Fake = 1
    """
    files = {
        "gossipcop_real.csv": 0,
        "gossipcop_fake.csv": 1,
        "politifact_real.csv": 0,
        "politifact_fake.csv": 1
    }
    
    df_list = []
    for file_name, label in files.items():
        file_path = os.path.join(dataset_dir, file_name)
        if os.path.exists(file_path):
            print(f"Loading {file_name}...")
            df = pd.read_csv(file_path)
            df['text'] = df['title'].apply(clean_text)
            df['label'] = label
            df = df[['text', 'label', 'title']]
            df_list.append(df)
        else:
            print(f"Warning: Dataset file not found at {file_path}")
            
    if not df_list:
        raise FileNotFoundError(f"No dataset CSV files found in {dataset_dir}!")
        
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df[combined_df['text'] != '']
    print(f"Total combined dataset size: {len(combined_df)}")
    print(f"Class distribution:\n{combined_df['label'].value_counts()}")
    
    return combined_df

def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Performs stratified split on the dataset into Train, Val, and Test dataframes.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Splits must sum to 1.0"
    
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df, 
        test_size=temp_ratio, 
        stratify=df['label'].values, 
        random_state=random_seed
    )
    
    val_relative_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative_ratio),
        stratify=temp_df['label'].values,
        random_state=random_seed
    )
    
    print(f"Dataset Split summary:")
    print(f"  Train: {len(train_df)} samples (Real={np.sum(train_df['label']==0)}, Fake={np.sum(train_df['label']==1)})")
    print(f"  Val:   {len(val_df)} samples (Real={np.sum(val_df['label']==0)}, Fake={np.sum(val_df['label']==1)})")
    print(f"  Test:  {len(test_df)} samples (Real={np.sum(test_df['label']==0)}, Fake={np.sum(test_df['label']==1)})")
    
    return train_df, val_df, test_df

def get_class_weights(labels):
    """
    Computes class weights for cross-entropy loss to handle imbalance.
    """
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return weights

def main():
    parser = argparse.ArgumentParser(description="Tiền xử lý và chia tập dữ liệu phân loại tin giả")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Thư mục chứa các file dataset CSV gốc")
    parser.add_argument("--output_dir", type=str, default="data_result", help="Thư mục để lưu các file đã xử lý")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    if args.dataset_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.dataset_dir = os.path.join(os.path.dirname(script_dir), "dataset")
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Bắt đầu tiền xử lý dữ liệu từ: {args.dataset_dir}")
    df = load_and_combine_data(args.dataset_dir)
    train_df, val_df, test_df = split_data(df, random_seed=args.seed)
    
    # Save to CSV files in data_result
    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Đã lưu các tập dữ liệu tiền xử lý:")
    print(f"  - Train: {train_path}")
    print(f"  - Val:   {val_path}")
    print(f"  - Test:  {test_path}")
    
    weights = get_class_weights(train_df['label'].values)
    print(f"Trọng số lớp (Class weights - balanced) trên tập Train: {weights}")
    print("Tiền xử lý hoàn tất thành công!")

if __name__ == "__main__":
    main()
