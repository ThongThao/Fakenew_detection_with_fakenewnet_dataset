import os
import pickle
import argparse
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast

# Import components
from preprocessing import clean_text
from lstm_model import Vocab, LSTMClassifier
from transformer_model import get_distilbert_model
from eval import load_best_model_config

# Colors for premium terminal display
GREEN = "\033[1;32m"
RED = "\033[1;31m"
BLUE = "\033[1;34m"
YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
RESET = "\033[0m"

def load_lstm_predictor(output_dir, device):
    """
    Locates the best LSTM checkpoint, loads its vocabulary, reinstantiates the model, and loads weights.
    Loads from output_dir/lstm subfolder.
    """
    try:
        best_config = load_best_model_config(output_dir, "lstm")
        param_suffix = f"lstm_lr{best_config['learning_rate']}_drop{best_config['dropout']}_bs{best_config['batch_size']}"
        
        lstm_folder = os.path.join(output_dir, "lstm")
        
        # Load vocab
        vocab_path = os.path.join(lstm_folder, f"vocab_{param_suffix}.pkl")
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
            
        # Instantiate model
        model = LSTMClassifier(
            vocab_size=len(vocab),
            dropout_rate=best_config['dropout']
        ).to(device)
        
        # Load weights
        checkpoint_path = os.path.join(lstm_folder, f"checkpoint_{param_suffix}.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        return model, vocab, best_config
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        return None, None, None

def load_distilbert_predictor(output_dir, transformer_model_name, device):
    """
    Locates the best DistilBERT checkpoint, loads its tokenizer, reinstantiates the model, and loads weights.
    Loads from output_dir/distilbert subfolder.
    """
    try:
        best_config = load_best_model_config(output_dir, "distilbert")
        param_suffix = f"distilbert_lr{best_config['learning_rate']}_drop{best_config['dropout']}_bs{best_config['batch_size']}"
        
        bert_folder = os.path.join(output_dir, "distilbert")
        
        # Load tokenizer
        tokenizer = DistilBertTokenizerFast.from_pretrained(transformer_model_name)
        
        # Instantiate model
        model = get_distilbert_model(transformer_model_name, dropout_rate=best_config['dropout']).to(device)
        
        # Load weights
        checkpoint_path = os.path.join(bert_folder, f"checkpoint_{param_suffix}.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        return model, tokenizer, best_config
    except Exception as e:
        print(f"Error loading DistilBERT model: {e}")
        return None, None, None

def predict_lstm(model, vocab, text, device, max_len=50):
    cleaned = clean_text(text)
    encoded = vocab.encode(cleaned)
    
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded = encoded + [vocab.w2i[vocab.pad_token]] * (max_len - len(encoded))
        
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = torch.argmax(logits, dim=1).item()
        
    return pred_label, probs

def predict_distilbert(model, tokenizer, text, device, max_len=64):
    cleaned = clean_text(text)
    
    encoding = tokenizer(
        cleaned,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = torch.argmax(logits, dim=1).item()
        
    return pred_label, probs

def display_prediction(title, label, probs, model_name):
    print(f"\n{CYAN}--- Kết quả từ mô hình {model_name.upper()} ---{RESET}")
    print(f"Tiêu đề: {YELLOW}\"{title}\"{RESET}")
    
    prob_real = probs[0] * 100
    prob_fake = probs[1] * 100
    
    if label == 1:
        print(f"Phân loại: {RED}TIN GIẢ (FAKE){RESET}")
        print(f"Độ tin cậy: {RED}{prob_fake:.2f}%{RESET} Fake (Real: {prob_real:.2f}%)")
    else:
        print(f"Phân loại: {GREEN}TIN THẬT (REAL){RESET}")
        print(f"Độ tin cậy: {GREEN}{prob_real:.2f}%{RESET} Real (Fake: {prob_fake:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Demo dự đoán tin thật/tin giả")
    parser.add_argument("--text", type=str, help="Văn bản cần kiểm tra phân loại")
    parser.add_argument("--output_dir", type=str, default="result", help="Thư mục gốc chứa kết quả huấn luyện")
    parser.add_argument("--transformer_model", type=str, default="distilbert-base-uncased", help="Tên model DistilBERT")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading best models from '{args.output_dir}'...")
    
    lstm_model, lstm_vocab, lstm_config = load_lstm_predictor(args.output_dir, device)
    bert_model, bert_tokenizer, bert_config = load_distilbert_predictor(args.output_dir, args.transformer_model, device)
    
    if lstm_model is None and bert_model is None:
        print(f"\n{RED}Error: Không thể tải mô hình nào. Vui lòng kiểm tra xem bạn đã huấn luyện mô hình chưa.{RESET}")
        return
        
    if args.text:
        text = args.text
        if lstm_model:
            label, probs = predict_lstm(lstm_model, lstm_vocab, text, device)
            display_prediction(text, label, probs, "lstm")
        if bert_model:
            label, probs = predict_distilbert(bert_model, bert_tokenizer, text, device)
            display_prediction(text, label, probs, "distilbert")
    else:
        print(f"\n{BLUE}=== CHƯƠNG TRÌNH PHÁT HIỆN TIN GIẢ (FAKE NEWS DETECTION DEMO) ==={RESET}")
        print("Tập dữ liệu huấn luyện: FakeNewsNet (English). Để đạt kết quả tốt nhất, hãy thử các câu tiếng Anh.")
        print("Có thể sử dụng cả câu tiếng Việt do DistilBERT/LSTM cũng hỗ trợ các ký tự Unicode.\n")
        
        while True:
            try:
                text = input(f"\nNhập tiêu đề tin tức cần kiểm tra (hoặc gõ '{YELLOW}exit{RESET}' để thoát): ").strip()
                if not text:
                    continue
                if text.lower() == 'exit':
                    break
                    
                if lstm_model:
                    label, probs = predict_lstm(lstm_model, lstm_vocab, text, device)
                    display_prediction(text, label, probs, "lstm")
                    
                if bert_model:
                    label, probs = predict_distilbert(bert_model, bert_tokenizer, text, device)
                    display_prediction(text, label, probs, "distilbert")
                    
            except KeyboardInterrupt:
                break
                
        print(f"\n{BLUE}Tạm biệt! Cảm ơn bạn đã trải nghiệm.{RESET}")

if __name__ == "__main__":
    main()
