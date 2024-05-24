import torch
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Example data for sentiment analysis
texts = ["I love this movie!", "This movie is terrible."]
labels = [1, 0]  # 1 for positive sentiment, 0 for negative sentiment

# Tokenize input texts
tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Prepare input tensors
input_ids = tokenized_texts["input_ids"]
attention_masks = tokenized_texts["attention_mask"]

# Define dataset
dataset = TensorDataset(input_ids, attention_masks, torch.tensor(labels))

# Define model with adapter layers
class SentimentClassifier(nn.Module):
    def __init__(self, bert_model):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.adapter = nn.Linear(768, 1)  # Adapter layer with one output for binary sentiment classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs[1]  # Take the pooled output (CLS token representation)
        sentiment_logits = self.adapter(pooled_output)
        sentiment_probs = self.sigmoid(sentiment_logits)
        return sentiment_probs.view(-1)

# Instantiate the model
model = SentimentClassifier(bert_model)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Define loss function
loss_fn = nn.BCELoss()

# Training loop
epochs = 3
batch_size = 2

for epoch in range(epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids_batch, attention_masks_batch, labels_batch = batch
        sentiment_probs = model(input_ids_batch, attention_masks_batch)
        loss = loss_fn(sentiment_probs, labels_batch.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# Example inference
test_text = "This movie is fantastic!"
test_input = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
test_input_ids = test_input["input_ids"]
test_attention_mask = test_input["attention_mask"]
predicted_sentiment = model(test_input_ids, test_attention_mask).item()
print(f"Predicted sentiment probability: {predicted_sentiment:.4f}")
