import torch
from torch.utils.data import Dataset
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

class TransformerDataset(Dataset):
    """
    PyTorch Dataset for Transformer models (DistilBERT).
    Uses the Hugging Face tokenizer to generate input_ids and attention_mask.
    """
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_distilbert_model(model_name="distilbert-base-uncased", dropout_rate=0.3):
    """
    Loads pretrained DistilBERT for classification, overriding default dropouts.
    """
    print(f"Loading {model_name} with dropout={dropout_rate}...")
    
    # Load configuration first and update the dropout rates
    config = DistilBertConfig.from_pretrained(model_name)
    config.dropout = dropout_rate
    config.attention_dropout = dropout_rate
    config.seq_classif_dropout = dropout_rate
    config.num_labels = 2
    
    # Load model with this configuration
    model = DistilBertForSequenceClassification.from_pretrained(model_name, config=config)
    return model
