import torch
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder, \
                         DPRContextEncoderTokenizer, DPRContextEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_question_dpr():
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    # tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    # model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    embedding_dimension = model.config.hidden_size
    return tokenizer, model, embedding_dimension

def load_context_dpr():
    tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    # tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    # model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    embedding_dimension = model.config.hidden_size
    return tokenizer, model, embedding_dimension

class DPRTextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def get_dpr_embeddings_batched(tokenizer, model, texts, batch_size=32, show_progress_bar=True, device='cuda'):
    model.to(device)
    model.eval() 
    dataset = DPRTextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings = []
    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc="Procesando dpr...", unit="batch", disable=not show_progress_bar):
            max_model_input_length = model.config.max_position_embeddings
            if max_model_input_length is None:
                max_model_input_length = 512 
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_model_input_length,
                return_tensors='pt'
            ).to(device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

    