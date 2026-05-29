import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Try importing plotting libraries, fallback gracefully if unavailable
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Charts will not be generated.")

# Import modular components
from lstm_model import Vocab, LSTMDataset, LSTMClassifier
from transformer_model import TransformerDataset, get_distilbert_model
from transformers import DistilBertTokenizerFast

def load_best_model_config(output_dir, model_type):
    """
    Reads the global results CSV to find the best configuration (based on Validation F1-score) for a model type.
    """
    csv_path = os.path.join(output_dir, "all_runs_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found at {csv_path}. Please train your models first!")
        
    df = pd.read_csv(csv_path)
    df_model = df[df['model'] == model_type]
    
    if df_model.empty:
        raise ValueError(f"No trained configurations found for model type: {model_type} in {csv_path}")
        
    # Sort by val_f1 descending
    best_row = df_model.sort_values(by='val_f1', ascending=False).iloc[0]
    return best_row

def evaluate_and_get_predictions(model, dataloader, device, is_transformer=False):
    """
    Runs model inference to return ground truths, predicted labels, and probabilities of class 1 (Fake).
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
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
                
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(cm, classes, title, save_path):
    """
    Draws and saves a confusion matrix without external dependencies other than matplotlib.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    # Style formatting
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12, fontweight='bold')

    plt.ylabel('Ground Truth Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_history_curves(history_data, model_name, save_loss_path, save_f1_path):
    """
    Plots the training and validation Loss and F1 curves over epochs.
    """
    epochs = [h['epoch'] for h in history_data]
    train_loss = [h['train_loss'] for h in history_data]
    val_loss = [h['val_loss'] for h in history_data]
    train_f1 = [h['train_f1'] for h in history_data]
    val_f1 = [h['val_f1'] for h in history_data]
    
    # 1. Loss Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_loss, 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs, val_loss, 's-', label='Val Loss', color='#ff7f0e', linewidth=2)
    plt.title(f'Loss Curve - {model_name.upper()}', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(save_loss_path, dpi=150)
    plt.close()
    
    # 2. F1 Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_f1, 'o-', label='Train F1 (Macro)', color='#2ca02c', linewidth=2)
    plt.plot(epochs, val_f1, 's-', label='Val F1 (Macro)', color='#d62728', linewidth=2)
    plt.title(f'Validation F1 Curve - {model_name.upper()}', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('F1 Score', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(save_f1_path, dpi=150)
    plt.close()

def plot_comparison_bar(lstm_metrics, bert_metrics, save_path):
    """
    Draws a grouped bar chart comparing performance metrics of both models.
    """
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    lstm_vals = [lstm_metrics['test_accuracy'], lstm_metrics['test_precision'], lstm_metrics['test_recall'], lstm_metrics['test_f1']]
    bert_vals = [bert_metrics['test_accuracy'], bert_metrics['test_precision'], bert_metrics['test_recall'], bert_metrics['test_f1']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, lstm_vals, width, label='LSTM (Best)', color='#4682B4')
    rects2 = ax.bar(x + width/2, bert_vals, width, label='DistilBERT (Best)', color='#FF8C00')
    
    ax.set_ylabel('Scores', fontsize=11, fontweight='bold')
    ax.set_title('Test Set Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình tốt nhất và tạo biểu đồ so sánh")
    parser.add_argument("--output_dir", type=str, default="result", help="Thư mục chứa kết quả huấn luyện")
    parser.add_argument("--data_dir", type=str, default="data_result", help="Thư mục chứa file test.csv đã xử lý")
    parser.add_argument("--transformer_model", type=str, default="distilbert-base-uncased", help="Tên model DistilBERT từ Hugging Face")
    parser.add_argument("--subset_ratio", type=float, default=None, help="Tỷ lệ lấy mẫu nhỏ để chạy nhanh (0.01 - 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    charts_dir = os.path.join(args.output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Load Preprocessed Test Data
    test_path = os.path.join(args.data_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Không tìm thấy file '{test_path}'. "
            f"Vui lòng chạy tiền xử lý trước: python fakenews_detection/preprocessing.py --output_dir {args.data_dir}"
        )
        
    print(f"Đang tải dữ liệu kiểm thử từ {test_path}...")
    test_df = pd.read_csv(test_path).dropna(subset=['text'])
    if args.subset_ratio is not None and 0.0 < args.subset_ratio < 1.0:
        test_df = test_df.sample(frac=args.subset_ratio, random_state=args.seed).reset_index(drop=True)
        
    test_texts, test_labels = test_df['text'].values, test_df['label'].values
    print(f"Loaded {len(test_texts)} test samples.")
    
    best_models = {}
    
    # Evaluate Best LSTM & Best DistilBERT
    for model_type in ["lstm", "distilbert"]:
        print(f"\n=================== Loading best {model_type.upper()} model ===================")
        try:
            best_config = load_best_model_config(args.output_dir, model_type)
            print(f"Best configuration found:")
            print(f"  Learning Rate: {best_config['learning_rate']}")
            print(f"  Dropout:       {best_config['dropout']}")
            print(f"  Batch Size:    {best_config['batch_size']}")
            print(f"  Val F1-Score:  {best_config['val_f1']:.4f}")
            print(f"  Test Accuracy: {best_config['test_accuracy']:.4f}")
            
            param_suffix = f"{model_type}_lr{best_config['learning_rate']}_drop{best_config['dropout']}_bs{best_config['batch_size']}"
            
            # Look inside model subfolder
            model_subfolder = os.path.join(args.output_dir, model_type)
            checkpoint_file = os.path.join(model_subfolder, f"checkpoint_{param_suffix}.pt")
            
            if model_type == "lstm":
                vocab_path = os.path.join(model_subfolder, f"vocab_{param_suffix}.pkl")
                with open(vocab_path, 'rb') as f:
                    vocab = pickle.load(f)
                
                test_dataset = LSTMDataset(test_texts, test_labels, vocab)
                model = LSTMClassifier(
                    vocab_size=len(vocab),
                    dropout_rate=best_config['dropout']
                ).to(device)
            else:
                tokenizer = DistilBertTokenizerFast.from_pretrained(args.transformer_model)
                test_dataset = TransformerDataset(test_texts, test_labels, tokenizer)
                model = get_distilbert_model(args.transformer_model, dropout_rate=best_config['dropout']).to(device)
                
            model.load_state_dict(torch.load(checkpoint_file, map_location=device))
            test_loader = DataLoader(test_dataset, batch_size=int(best_config['batch_size']), shuffle=False)
            
            is_transformer = (model_type == "distilbert")
            y_true, y_pred, y_probs = evaluate_and_get_predictions(model, test_loader, device, is_transformer)
            
            best_models[model_type] = {
                'config': best_config,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_probs': y_probs,
                'suffix': param_suffix
            }
            
            print(f"\nClassification Report for {model_type.upper()}:")
            print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
            
            cm = confusion_matrix(y_true, y_pred)
            print(f"Confusion Matrix for {model_type.upper()}:")
            print(cm)
            
            if HAS_MATPLOTLIB:
                plot_confusion_matrix(
                    cm, 
                    classes=["Real", "Fake"], 
                    title=f"Confusion Matrix - {model_type.upper()}", 
                    save_path=os.path.join(charts_dir, f"confusion_matrix_{model_type}.png")
                )
                print(f"Saved confusion matrix chart to {os.path.join(charts_dir, f'confusion_matrix_{model_type}.png')}")
                
                history_path = os.path.join(model_subfolder, f"history_{param_suffix}.json")
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        history_data = json.load(f)
                    plot_history_curves(
                        history_data,
                        model_type,
                        save_loss_path=os.path.join(charts_dir, f"loss_curve_{model_type}.png"),
                        save_f1_path=os.path.join(charts_dir, f"f1_curve_{model_type}.png")
                    )
                    print(f"Saved history curves to {charts_dir}")
                
        except Exception as e:
            print(f"Could not evaluate {model_type.upper()} model: {e}")
            
    if HAS_MATPLOTLIB and len(best_models) == 2:
        print("\n=================== Generating Comparative Metrics Charts ===================")
        lstm_metrics = best_models["lstm"]["config"]
        bert_metrics = best_models["distilbert"]["config"]
        
        plot_comparison_bar(
            lstm_metrics,
            bert_metrics,
            save_path=os.path.join(charts_dir, "model_comparison_metrics.png")
        )
        print(f"Saved comparison bar chart to {os.path.join(charts_dir, 'model_comparison_metrics.png')}")
        
        conclusion_path = os.path.join(args.output_dir, "best_models_conclusion.json")
        conclusions = {
            "best_lstm": {
                "lr": float(lstm_metrics["learning_rate"]),
                "dropout": float(lstm_metrics["dropout"]),
                "batch_size": int(lstm_metrics["batch_size"]),
                "test_accuracy": float(lstm_metrics["test_accuracy"]),
                "test_precision": float(lstm_metrics["test_precision"]),
                "test_recall": float(lstm_metrics["test_recall"]),
                "test_f1": float(lstm_metrics["test_f1"])
            },
            "best_distilbert": {
                "lr": float(bert_metrics["learning_rate"]),
                "dropout": float(bert_metrics["dropout"]),
                "batch_size": int(bert_metrics["batch_size"]),
                "test_accuracy": float(bert_metrics["test_accuracy"]),
                "test_precision": float(bert_metrics["test_precision"]),
                "test_recall": float(bert_metrics["test_recall"]),
                "test_f1": float(bert_metrics["test_f1"])
            },
            "winner": "distilbert" if bert_metrics["test_f1"] > lstm_metrics["test_f1"] else "lstm"
        }
        with open(conclusion_path, 'w') as f:
            json.dump(conclusions, f, indent=4)
        print(f"Saved best models comparison conclusions to {conclusion_path}")

if __name__ == "__main__":
    main()
