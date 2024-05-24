import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Example data
texts = ["I love this movie!", "This movie is terrible."]
labels = torch.tensor([1, 0], dtype=torch.float32)  # 1 for positive, 0 for negative

# Tokenize input texts
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Define a simple model with an adapter layer
class SentimentClassifier(nn.Module):
    def __init__(self, bert_model):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.adapter = nn.Linear(768, 1)  # Adapter layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        sentiment_logits = self.adapter(pooled_output)
        sentiment_probs = self.sigmoid(sentiment_logits)
        return sentiment_probs

# Instantiate the model
model = SentimentClassifier(bert_model)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.adapter.parameters(), lr=1e-4)  # Only train the adapter

# Training loop (simplified)
model.train()
input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
optimizer.zero_grad()
outputs = model(input_ids, attention_mask).squeeze()
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
print(f"Loss: {loss.item():.4f}")

# Example inference
model.eval()
test_text = "This movie is fantastic!"
test_input = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    test_output = model(test_input["input_ids"], test_input["attention_mask"]).item()
print(f"Predicted sentiment probability: {test_output:.4f}")
