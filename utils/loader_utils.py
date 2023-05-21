import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class EssayDataset(Dataset):
    def __init__(self, df, config) -> None:
        tokenizer = AutoTokenizer.from_pretrained(config.lm_path)

        self.essay_id = df['essay_id'].tolist()
        self.essay = df['essay'].tolist()
        self.essay_set = torch.LongTensor(df['essay_set'].tolist()).cuda() 
        self.score =  torch.LongTensor(df['score'].tolist()).cuda()
        self.score_scaled = torch.FloatTensor(df['score_scaled'].tolist()).cuda()

        self.tokenized_essay = tokenizer(
            self.essay,
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            max_length=config.seq_len,
            return_tensors="pt",
        ).to('cuda')

    def __len__(self):
        return len(self.essay)
    
    def __getitem__(self, index):

        return self.tokenized_essay['input_ids'][index], self.tokenized_essay['token_type_ids'][index],\
        self.tokenized_essay['attention_mask'][index], self.score[index], \
        self.score_scaled[index], self.essay_set[index] 
    

def collate_funcion(batch):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    score = []
    score_scaled = []
    essay_set = []
    for input_ids_, token_type_ids_, attention_mask_, score_, score_scaled_, essay_set_ in batch:
        input_ids.append(input_ids_)
        token_type_ids.append(token_type_ids_)
        attention_mask.append(attention_mask_)
        score.append(score_)
        score_scaled.append(score_scaled_)
        essay_set.append(essay_set_)

    batch = {
        'input_ids': torch.stack(input_ids, dim=0),
        'token_type_ids': torch.stack(token_type_ids, dim=0),
        'attention_mask': torch.stack(attention_mask, dim=0),
        'score': torch.stack(score, dim=0),
        'score_scaled': torch.stack(score_scaled, dim=0),
        'essay_set': torch.stack(essay_set, dim=0),
    }

    return batch   

        

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)