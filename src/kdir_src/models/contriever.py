import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_contriever():
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')
    embedding_dimension = model.config.hidden_size
    return tokenizer, model, embedding_dimension

def load_contriever_ft():
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
    model = AutoModel.from_pretrained('facebook/contriever-msmarco')
    embedding_dimension = model.config.hidden_size
    return tokenizer, model, embedding_dimension

class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

def get_contriever_embeddings_batched(tokenizer, model, sentences, batch_size=32, show_progress_bar=True, device='cuda'):
    model.to(device)
    model.eval()
    dataset = SentenceDataset(sentences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    with torch.no_grad():
        for batch_sentences in tqdm(dataloader, desc="Procesando contriever...", unit="batch", disable= not show_progress_bar):
            inputs = tokenizer(
                batch_sentences, 
                padding=True, 
                truncation=True, 
                return_tensors='pt' 
            ).to(device)
            
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

