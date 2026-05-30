import os
import time
import pickle
import json
import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import local models and preprocessing
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import clean_text
from lstm_model import Vocab, LSTMClassifier
from transformer_model import get_distilbert_model
from transformers import DistilBertTokenizerFast

# Initialize FastAPI
app = FastAPI(title="Fake News Detection API", description="LSTM vs DistilBERT Fake News Predictor")

# Resolve absolute paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)  # d:\FakeNewsNet_Detection
STATIC_DIR = os.path.join(CURRENT_DIR, "static")

# Configuration for the best models
LSTM_CHECKPOINT = os.path.join(BASE_DIR, "result", "checkpoint", "checkpoint_lstm_lr0.001_drop0.1_bs8.pt")
LSTM_VOCAB_PATH = os.path.join(BASE_DIR, "result", "lstm", "vocab_lstm_lr0.001_drop0.1_bs8.pkl")
BERT_CHECKPOINT = os.path.join(BASE_DIR, "result", "checkpoint", "checkpoint_distilbert_lr5e-05_drop0.3_bs8.pt")
BERT_MODEL_NAME = "distilbert-base-uncased"

ALL_RUNS_CSV = os.path.join(BASE_DIR, "result", "all_runs_results.csv")
CONCLUSIONS_JSON = os.path.join(BASE_DIR, "result", "best_models_conclusion.json")
LSTM_BEST_HISTORY = os.path.join(BASE_DIR, "result", "lstm", "history_lstm_lr0.001_drop0.1_bs8.json")
BERT_BEST_HISTORY = os.path.join(BASE_DIR, "result", "distilbert", "history_distilbert_lr5e-05_drop0.3_bs8.json")

# Global variables for loaded models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = None
lstm_vocab = None
bert_model = None
bert_tokenizer = None

# Input schemas
class PredictRequest(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    global lstm_model, lstm_vocab, bert_model, bert_tokenizer
    print(f"Using device: {device}")
    
    # 1. Load LSTM model & vocab
    try:
        if os.path.exists(LSTM_VOCAB_PATH):
            with open(LSTM_VOCAB_PATH, 'rb') as f:
                lstm_vocab = pickle.load(f)
            print("Loaded LSTM vocabulary successfully.")
        else:
            print(f"Warning: Vocab file not found at {LSTM_VOCAB_PATH}")
            
        if os.path.exists(LSTM_CHECKPOINT) and lstm_vocab is not None:
            lstm_model = LSTMClassifier(
                vocab_size=len(lstm_vocab),
                dropout_rate=0.1  # Matches best model configuration
            )
            lstm_model.load_state_dict(torch.load(LSTM_CHECKPOINT, map_location=device))
            lstm_model.to(device)
            lstm_model.eval()
            print("Loaded best LSTM model weights successfully.")
        else:
            print(f"Warning: LSTM checkpoint or vocab missing.")
    except Exception as e:
        print(f"Error loading LSTM: {e}")

    # 2. Load DistilBERT model & tokenizer
    try:
        if os.path.exists(BERT_CHECKPOINT):
            # Load tokenizer
            bert_tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
            # Recreate model with best dropout
            bert_model = get_distilbert_model(BERT_MODEL_NAME, dropout_rate=0.3)
            bert_model.load_state_dict(torch.load(BERT_CHECKPOINT, map_location=device))
            bert_model.to(device)
            bert_model.eval()
            print("Loaded best DistilBERT model weights successfully.")
        else:
            print(f"Warning: DistilBERT checkpoint missing at {BERT_CHECKPOINT}")
    except Exception as e:
        print(f"Error loading DistilBERT: {e}")

@app.post("/api/predict")
async def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
        
    cleaned = clean_text(req.text)
    
    results = {}
    
    # LSTM Prediction
    if lstm_model is not None and lstm_vocab is not None:
        start_t = time.perf_counter()
        encoded = lstm_vocab.encode(cleaned)
        max_len = 50
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        else:
            encoded = encoded + [lstm_vocab.w2i[lstm_vocab.pad_token]] * (max_len - len(encoded))
            
        input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = lstm_model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_label = int(torch.argmax(logits, dim=1).item())
            
        elapsed = (time.perf_counter() - start_t) * 1000 # ms
        results["lstm"] = {
            "label": "Fake" if pred_label == 1 else "Real",
            "prob_real": float(probs[0]),
            "prob_fake": float(probs[1]),
            "latency_ms": round(elapsed, 2)
        }
    else:
        results["lstm"] = {"error": "LSTM model not loaded"}
        
    # DistilBERT Prediction
    if bert_model is not None and bert_tokenizer is not None:
        start_t = time.perf_counter()
        encoding = bert_tokenizer(
            cleaned,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_label = int(torch.argmax(logits, dim=1).item())
            
        elapsed = (time.perf_counter() - start_t) * 1000 # ms
        results["distilbert"] = {
            "label": "Fake" if pred_label == 1 else "Real",
            "prob_real": float(probs[0]),
            "prob_fake": float(probs[1]),
            "latency_ms": round(elapsed, 2)
        }
    else:
        results["distilbert"] = {"error": "DistilBERT model not loaded"}
        
    results["device"] = str(device)
    return results

@app.get("/api/history")
async def get_history():
    if not os.path.exists(ALL_RUNS_CSV):
        raise HTTPException(status_code=404, detail="all_runs_results.csv not found")
    try:
        df = pd.read_csv(ALL_RUNS_CSV)
        # Fill NaN values to avoid JSON issues
        df = df.fillna(0)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/best_conclusions")
async def get_conclusions():
    if not os.path.exists(CONCLUSIONS_JSON):
        raise HTTPException(status_code=404, detail="best_models_conclusion.json not found")
    try:
        with open(CONCLUSIONS_JSON, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/curves")
async def get_curves():
    results = {}
    
    # Load LSTM history
    if os.path.exists(LSTM_BEST_HISTORY):
        try:
            with open(LSTM_BEST_HISTORY, 'r') as f:
                results["lstm"] = json.load(f)
        except Exception as e:
            results["lstm_error"] = str(e)
    else:
        results["lstm_error"] = "File not found"
        
    # Load DistilBERT history
    if os.path.exists(BERT_BEST_HISTORY):
        try:
            with open(BERT_BEST_HISTORY, 'r') as f:
                results["distilbert"] = json.load(f)
        except Exception as e:
            results["distilbert_error"] = str(e)
    else:
        results["distilbert_error"] = "File not found"
        
    return results

# Create directory for static files if it doesn't exist
os.makedirs(STATIC_DIR, exist_ok=True)

# Serve Frontend static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Server is running, but index.html is missing inside static/ folder."}
