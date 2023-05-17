from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import string

def generate_random_string(length):
    """Generate a random string of specified length."""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))
sentences=[]
# Load data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
for i in  range(10):
    sentences.append('This is the first sentence' + generate_random_string(1000))
input_ids = []
attention_masks = []
for sentence in sentences:
    encoded_dict = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=512,
        # padding= 'max_length',
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
dataset = TensorDataset(input_ids, attention_masks)
dataloader = DataLoader(dataset, batch_size=3)

# Load model
model = BertModel.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_masks).last_hidden_state
    print(outputs.shape)
# embeddings = outputs.last_hidden_state
# print(embeddings.shape)
# # Compute embeddings for each batch
# for batch in dataloader:
#     batch_input_ids = batch[0]
#     batch_attention_masks = batch[1]
#     outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
#     embeddings = outputs.last_hidden_state
#     print(embeddings.shape)
