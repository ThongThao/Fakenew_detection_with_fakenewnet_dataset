# FakeNewsNet Detection & Evaluation Portal

This repository contains a complete pipeline for **Fake News Detection** trained on the **FakeNewsNet** dataset (including PolitiFact and GossipCop articles). It includes text preprocessing, model training pipelines (Bi-LSTM vs. DistilBERT), model evaluation scripts, and an interactive FastAPI-based web application to test news authenticity and visualize training performance metrics.

---

## 🌟 Key Features

1. **Robust Text Preprocessing**: Cleans HTML tags, URLs, and extra whitespaces while preserving multilingual characters and applying WordNet lemmatizers.
2. **Model Training Grid Search**: Supports training and hyperparameter tuning for:
   - **Bi-LSTM** (Bidirectional LSTM Classifier with GloVe/Embedding representations).
   - **DistilBERT** (Hugging Face transformer classification).
   - Dynamic parameter tuning across different Learning Rates, Dropouts (`0.1 / 0.3 / 0.5`), and Batch Sizes (`8 / 16 / 32`).
3. **Automated Evaluation**: Evaluates best-performing models on the test set, creating confusion matrices, learning history curves, and overall summaries.
4. **Interactive FastAPI Web Dashboard**:
   - **Real-time Predictor**: Evaluates user input news side-by-side using the best LSTM and DistilBERT models.
   - **Comparative Analytics**: Shows performance charts (Accuracy, Precision, Recall, F1) and training progression curves.
   - **Hyperparameter Grid**: Interactive table displaying performance data for all 36 training configurations.

---

## 📂 Project Structure

```bash
├── dataset/                        # Raw CSV articles from Politifact and GossipCop
├── data_result/                    # Split CSV datasets (train.csv, val.csv, test.csv)
├── result/                         # Training outputs, metrics, and models
│   ├── checkpoint/                 # Saved model weights (.pt files)
│   ├── lstm/                       # LSTM validation history logs & pickle vocab files
│   ├── distilbert/                 # DistilBERT validation history logs
│   ├── charts/                     # Generated evaluation curve charts
│   ├── all_runs_results.csv        # Comprehensive grid search results CSV
│   └── best_models_conclusion.json # Conclusion summary indicating the winner model
└── fakenews_detection/             # Core codebase
    ├── preprocessing.py            # Text cleaning and dataset division
    ├── lstm_model.py               # Custom vocab dataset, and LSTM PyTorch model
    ├── transformer_model.py        # Hugging Face DistilBERT configuration & model loader
    ├── train.py                    # Training & validation loop with Early Stopping
    ├── eval.py                     # Evaluation script generating test charts
    ├── app.py                      # FastAPI web server and API endpoints
    └── static/                     # Frontend dashboard assets
        ├── index.html              # Main HTML5 template
        ├── style.css               # Glassmorphic dark mode styling
        └── app.js                  # App state, Chart.js graphs, and UI bindings
```

---

## 🚀 Getting Started

### 1. Requirements & Setup

Make sure Python 3.8+ is installed on your system. Install the required dependencies:

```bash
pip install -r requirements.txt
pip install fastapi uvicorn pydantic transformers torch pandas scikit-learn
```

---

## 🧠 Machine Learning Pipeline

### Step 1: Preprocess Data
Run the preprocessing script to clean the articles and split them into Train, Validation, and Test sets:

```bash
python fakenews_detection/preprocessing.py
```
*Creates `train.csv`, `val.csv`, and `test.csv` in `data_result/`.* Detailed logic can be inspected in [preprocessing.py](file:///d:/FakeNewsNet_Detection/fakenews_detection/preprocessing.py).

### Step 2: Model Training
Train individual architectures with selected hyperparameters (Learning Rate, Dropout, and Batch Size) to run grid search optimization:

```bash
# Example training LSTM:
python fakenews_detection/train.py --model lstm --lr 0.001 --dropout 0.1 --batch_size 8 --epochs 10

# Example training DistilBERT:
python fakenews_detection/train.py --model distilbert --lr 5e-5 --dropout 0.3 --batch_size 8 --epochs 5
```
*Outputs checkpoints to `result/checkpoint/` and history logs to `result/<model>/`.* Detailed training logic is in [train.py](file:///d:/FakeNewsNet_Detection/fakenews_detection/train.py).

### Step 3: Run Model Evaluation
Analyze the performance of the best configurations and plot curves:

```bash
python fakenews_detection/eval.py
```
*Generates comparison charts in `result/charts/` and updates [best_models_conclusion.json](file:///d:/FakeNewsNet_Detection/result/best_models_conclusion.json).* See [eval.py](file:///d:/FakeNewsNet_Detection/fakenews_detection/eval.py).

---

## 💻 Running the Web Application

The interactive web portal acts as a dashboard to execute inferences and review model comparisons. Start the FastAPI backend:

```bash
uvicorn fakenews_detection.app:app --host 127.0.0.1 --port 8080
```

Once running, navigate to:
👉 **[http://127.0.0.1:8080/](http://127.0.0.1:8080/)**

### Dashboard Views
- **Nhận diện Tin tức (Predictor)**: Input any headline to classify it using the loaded checkpoints:
  - LSTM Checkpoint: [checkpoint_lstm_lr0.001_drop0.1_bs8.pt](file:///d:/FakeNewsNet_Detection/result/checkpoint/checkpoint_lstm_lr0.001_drop0.1_bs8.pt)
  - DistilBERT Checkpoint: [checkpoint_distilbert_lr5e-05_drop0.3_bs8.pt](file:///d:/FakeNewsNet_Detection/result/checkpoint/checkpoint_distilbert_lr5e-05_drop0.3_bs8.pt)
- **So sánh & Đánh giá (Analytics)**: View side-by-side bar charts of accuracy/F1 metrics, and line charts showing training history trends.
- **Bảng Hyperparameters (Tuning Grid)**: Render and query the grid of all 36 runs with filters for model, dropout rate, and batch size.

---

## 📁 Legacy Data Collector (Original FakeNewsNet)

To run the legacy Twitter scraper and crawler, refer to instructions inside the `code` directory:

1. Setup your Twitter API keys in `code/resources/tweet_keys_file.json`.
2. Configure settings inside `code/config.json`.
3. Launch the resource server and start collecting:
   ```bash
   cd code
   python -m resource_server.app
   python main.py
   ```

---

## 📚 References & Citations

If you use the FakeNewsNet dataset, please cite the following papers:

```bibtex
@article{shu2018fakenewsnet,
  title={FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media},
  author={Shu, Kai and Mahudeswaran, Deepak and Wang, Suhang and Lee, Dongwon and Liu, Huan},
  journal={arXiv preprint arXiv:1809.01286},
  year={2018}
}

@article{shu2017fake,
  title={Fake News Detection on Social Media: A Data Mining Perspective},
  author={Shu, Kai and Sliva, Amy and Wang, Suhang and Tang, Jiliang and Liu, Huan},
  journal={ACM SIGKDD Explorations Newsletter},
  volume={19},
  number={1},
  pages={22--36},
  year={2017},
  publisher={ACM}
}
```

---
*(C) 2026 FakeNewsNet Detection Portal.*
