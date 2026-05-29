import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Import our modular components
from preprocessing import get_class_weights
from lstm_model import Vocab, LSTMDataset, LSTMClassifier
from transformer_model import TransformerDataset, get_distilbert_model

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_pred):
    """
    Computes accuracy, precision, recall, and F1-score (macro and binary for class 1: Fake).
    """
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_fake, r_fake, f_fake, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': p_macro,
        'recall': r_macro,
        'f1': f_macro,
        'precision_fake': p_fake,
        'recall_fake': r_fake,
        'f1_fake': f_fake
    }

def train_epoch(model, dataloader, optimizer, criterion, device, is_transformer=False, scheduler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        labels = batch['label'].to(device)
        
        if is_transformer:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        else:
            input_ids = batch['input_ids'].to(device)
            logits = model(input_ids)
            
        loss = criterion(logits, labels)
        loss.backward()
        
        if is_transformer:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item() * len(labels)
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = epoch_loss
    return metrics

def eval_epoch(model, dataloader, criterion, device, is_transformer=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['label'].to(device)
            
            if is_transformer:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:
                input_ids = batch['input_ids'].to(device)
                logits = model(input_ids)
                
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = epoch_loss
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phát hiện tin giả")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "distilbert"], help="Chọn kiến trúc mô hình")
    parser.add_argument("--data_dir", type=str, default="data_result", help="Thư mục chứa train.csv, val.csv, test.csv đã xử lý")
    parser.add_argument("--lr", type=float, default=None, help="Tốc độ học (learning rate)")
    parser.add_argument("--dropout", type=float, default=0.3, choices=[0.1, 0.3, 0.5], help="Tỷ lệ dropout")
    parser.add_argument("--batch_size", type=int, default=16, choices=[8, 16, 32], help="Kích thước batch")
    parser.add_argument("--epochs", type=int, default=10, help="Số lượng epoch tối đa")
    parser.add_argument("--patience", type=int, default=3, help="Độ kiên nhẫn dừng sớm (Early stopping)")
    parser.add_argument("--output_dir", type=str, default="result", help="Thư mục gốc lưu kết quả (checkpoint lưu vào output_dir/<model>)")
    parser.add_argument("--transformer_model", type=str, default="distilbert-base-uncased", help="Tên model DistilBERT từ Hugging Face")
    parser.add_argument("--subset_ratio", type=float, default=None, help="Tỷ lệ lấy mẫu nhỏ để chạy nhanh (0.01 - 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    if args.lr is None:
        args.lr = 2e-5 if args.model == "distilbert" else 1e-3
        
    # Root output folders
    os.makedirs(args.output_dir, exist_ok=True)
    charts_dir = os.path.join(args.output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Model-specific subdirectory (result/lstm or result/distilbert)
    model_output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_output_dir, exist_ok=True)
    
    param_suffix = f"{args.model}_lr{args.lr}_drop{args.dropout}_bs{args.batch_size}"
    history_file = os.path.join(model_output_dir, f"history_{param_suffix}.json")
    checkpoint_file = os.path.join(model_output_dir, f"checkpoint_{param_suffix}.pt")
    
    # Load Preprocessed Data
    train_path = os.path.join(args.data_dir, "train.csv")
    val_path = os.path.join(args.data_dir, "val.csv")
    test_path = os.path.join(args.data_dir, "test.csv")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Không tìm thấy các tệp dữ liệu đã tiền xử lý tại '{args.data_dir}'. "
            f"Vui lòng chạy tiền xử lý trước: python fakenews_detection/preprocessing.py --output_dir {args.data_dir}"
        )
        
    print(f"Đang tải dữ liệu tiền xử lý từ {args.data_dir}...")
    train_df = pd.read_csv(train_path).dropna(subset=['text'])
    val_df = pd.read_csv(val_path).dropna(subset=['text'])
    test_df = pd.read_csv(test_path).dropna(subset=['text'])
    
    if args.subset_ratio is not None and 0.0 < args.subset_ratio < 1.0:
        print(f"Using only {args.subset_ratio * 100}% of the dataset for quick run...")
        train_df = train_df.sample(frac=args.subset_ratio, random_state=args.seed).reset_index(drop=True)
        val_df = val_df.sample(frac=args.subset_ratio, random_state=args.seed).reset_index(drop=True)
        test_df = test_df.sample(frac=args.subset_ratio, random_state=args.seed).reset_index(drop=True)
        
    train_texts, train_labels = train_df['text'].values, train_df['label'].values
    val_texts, val_labels = val_df['text'].values, val_df['label'].values
    test_texts, test_labels = test_df['text'].values, test_df['label'].values
    
    print(f"Loaded {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test samples.")
    
    class_weights = get_class_weights(train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Computed class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    is_transformer = (args.model == "distilbert")
    
    if args.model == "lstm":
        vocab = Vocab()
        vocab.build_vocab(train_texts)
        
        vocab_path = os.path.join(model_output_dir, f"vocab_{param_suffix}.pkl")
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Saved vocabulary to {vocab_path}")
        
        train_dataset = LSTMDataset(train_texts, train_labels, vocab)
        val_dataset = LSTMDataset(val_texts, val_labels, vocab)
        test_dataset = LSTMDataset(test_texts, test_labels, vocab)
        
        model = LSTMClassifier(
            vocab_size=len(vocab), 
            dropout_rate=args.dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = None
        
    else:
        tokenizer = DistilBertTokenizerFast.from_pretrained(args.transformer_model)
        
        train_dataset = TransformerDataset(train_texts, train_labels, tokenizer)
        val_dataset = TransformerDataset(val_texts, val_labels, tokenizer)
        test_dataset = TransformerDataset(test_texts, test_labels, tokenizer)
        
        model = get_distilbert_model(args.transformer_model, dropout_rate=args.dropout).to(device)
        
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(train_dataset) // args.batch_size * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps), 
            num_training_steps=total_steps
        )
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    best_val_f1 = -1
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    history = []
    
    print(f"\n--- Starting Training for {args.model.upper()} ---")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, is_transformer, scheduler)
        val_metrics = eval_epoch(model, val_loader, criterion, device, is_transformer)
        
        print(f"  [Train] Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1 (Macro): {train_metrics['f1']:.4f}")
        print(f"  [Val]   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1 (Macro): {val_metrics['f1']:.4f}")
        
        epoch_log = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1']
        }
        history.append(epoch_log)
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_file)
            print(f"  --> Checkpoint saved to {checkpoint_file} (Best Val F1: {best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement. Early stopping counter: {epochs_no_improve}/{args.patience}")
            
        if epochs_no_improve >= args.patience:
            print("Early stopping triggered! Training stopped to prevent overfitting.")
            break
            
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"\nEvaluating Best Model (from Epoch {best_epoch}) on Test Set...")
    model.load_state_dict(torch.load(checkpoint_file))
    test_metrics = eval_epoch(model, test_loader, criterion, device, is_transformer)
    print(f"  [Test] Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['accuracy']:.4f} | F1 (Macro): {test_metrics['f1']:.4f} | Recall: {test_metrics['recall']:.4f}")
    
    # Save training run summary to global CSV file in output_dir root
    results_csv_path = os.path.join(args.output_dir, "all_runs_results.csv")
    best_epoch_data = history[best_epoch - 1]
    
    new_result = {
        "model": args.model,
        "learning_rate": args.lr,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "train_loss": best_epoch_data["train_loss"],
        "train_accuracy": best_epoch_data["train_accuracy"],
        "train_precision": best_epoch_data["train_precision"],
        "train_recall": best_epoch_data["train_recall"],
        "train_f1": best_epoch_data["train_f1"],
        "val_loss": best_val_loss,
        "val_accuracy": best_epoch_data["val_accuracy"],
        "val_precision": best_epoch_data["val_precision"],
        "val_recall": best_epoch_data["val_recall"],
        "val_f1": best_val_f1,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"]
    }
    
    df_new = pd.DataFrame([new_result])
    if os.path.exists(results_csv_path):
        df_old = pd.read_csv(results_csv_path)
        mask = (df_old['model'] == args.model) & \
               (df_old['learning_rate'] == args.lr) & \
               (df_old['dropout'] == args.dropout) & \
               (df_old['batch_size'] == args.batch_size)
        df_old = df_old[~mask]
        df_results = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_results = df_new
        
    df_results.to_csv(results_csv_path, index=False)
    print(f"Successfully saved training run summary to {results_csv_path}")

if __name__ == "__main__":
    main()
