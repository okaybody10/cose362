import torch
import json
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
from kobert_transformers import get_tokenizer, get_kobert_model
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Cuda device check
devices = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# label preprocess
label_list = ['PS', 'FD', 'TR', 'AF', 'OG', 'LC', 'CV', 'DT', 'TI', 'TI', 'QT', 'EV', 'AM', 'PT', 'MT', "TM"] 
label_fin = []
label_fin += ['B-' + i for i in label_list]
label_fin += ['O']
label_fin += ['I-' + i for i in label_list]
label_to_idx = {label: idx for idx, label in enumerate(label_fin)}
idx_to_label = {idx: label for idx, label in enumerate(label_fin)}

def load_files(path='./dataset/NLNE2202211219.json') :
    print("===========Start Load files===========")
    with open(path, "r") as f :
        bef_data = json.load(f)
    print(f"Count number: {len(bef_data)}")
    print("===========End Load files===========")
    bef_data = bef_data['document']
    df_tot = pd.DataFrame(columns=['form', 'NE'])
    print("===========Start process Dataframe===========")
    for _, r in enumerate(tqdm(bef_data)) :
        df_tot = df_tot.append(pd.DataFrame.from_records(r['sentence'], columns=['form', 'NE']))
    df_tot.dropna(how='any', inplace=True)
    print("===========End process Dataframe===========")
    return df_tot

'''
We will return the label of given words, using the ne_lists
We use BIO-tagging
'''
def tagging(words: List[str], ne_lists: List[Dict[str, Any]]) -> List[str] :
    results = [i if i in ['[CLS]', '[SEP]', '[PAD]'] else 'O' for i in words] # If token is not Special, initialize 'O' tag
    ps_words = [i.replace('##', '').replace('▁','') for i in words]
    ne_cnt = len(ne_lists)
    ne_idx = -1
    ne_label = 0

    for idx, word in enumerate(ps_words) :
        if results[idx] != 'O' or word == '' or word == '[UNK]':
            continue
        if word == '[UNK]' :
            continue
        # Now condition check
        if ne_idx >= 0 : 
            nw_word = ne_lists[ne_idx]['form'][ne_label:]
        else :
            nw_word = ''

        # I-tag condition
        if (len(nw_word) > 0) & (nw_word.startswith(word)) & (results[idx-1][0] == 'B' or results[idx-1][0] == 'I') :
            results[idx] = 'I-' + ne_lists[ne_idx]['label'][:2]
            ne_label += len(word)
        else : # B-tag condition
            back_idx = ne_idx
            back_label = ne_label
            while ne_idx + 1 < ne_cnt :
                ne_idx += 1
                ne_label = 0
                nw_word = ne_lists[ne_idx]['form']
                if (len(nw_word) > 0) & (nw_word.startswith(word)) :
                    results[idx] = 'B-' + ne_lists[ne_idx]['label'][:2]
                    ne_label += len(word)
                    break
            if ne_idx + 1 == ne_cnt and ne_label == 0:
                ne_idx = back_idx
                ne_label = back_label
    O_idx = np.where(np.array(results)=="O")[0]
    drop_idx = np.random.choice(O_idx, size = int(np.round(0.3 * len(O_idx))))
    select_idx = np.delete(np.arange(len(results)), drop_idx) # select_idx
    results = [results[i] for i in select_idx]
    words = [words[i] for i in select_idx]
    return [words, results]

def collect_fn(batch) :
    tokenizer = get_tokenizer()
    model = get_kobert_model().to(devices)
    model.eval() # Only get embeddings
    convert_batch = {key: [i[key] for i in batch] for key in batch[0]}
    texts = convert_batch['texts']
    labels = convert_batch['labels']
    sentence = tokenizer(texts) # Input_ids, token_type_ids, attention_mask
    texts = []
    tags = []
    for (index, label) in zip(sentence['input_ids'], labels) :
        res = tagging(tokenizer.convert_ids_to_tokens(index), label)
        tags.append(res[1])
        texts.append(torch.tensor(tokenizer.convert_tokens_to_ids(res[0])))
    # tags = [tagging(tokenizer.convert_ids_to_tokens(index), label) for (index, label) in zip(sentence['input_ids'], labels)]
    # padding
    texts = pad_sequence(texts, batch_first= True, padding_value= 1).to(devices) # OK
    # tags = torch.where(tags == -1, '[PAD]', tags)
    # Embedding
    embeddings = model.embeddings.word_embeddings(texts) # (Batch, max_len, 768)
    # tags = tagging(tokenizer.convert_ids_to_tokens(sentence['input_ids']), self.labels[index])
    convert_tags = [torch.tensor([33 if i in ['[CLS]', '[SEP]', '[PAD]'] else label_to_idx[i] for i in tag]) for tag in tags] # (Batch, max_len)
    convert_tags = pad_sequence(convert_tags, batch_first=True, padding_value= 33).to(devices)
    return {
        'texts' : embeddings,
        'labels' : convert_tags
    } # Collect_fn?

class CustomDataset(Dataset) :
    def __init__(self, texts, labels) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) :
        return len(self.texts)
    
    def __getitem__(self, index) -> Any:
        # tokenizer
        return {
            'texts' : self.texts[index],
            'labels' : self.labels[index]
        }
        # input = self.texts[index]
        # sentence = self.tokenizer(input, padding = 'max_length', truncation = True, max_length = self.max_len) # Input_ids, token_type_ids, attention_mask
        # tags = tagging(self.tokenizer.convert_ids_to_tokens(sentence['input_ids']), self.labels[index])
        # convert_tags = [-100 if i in ['[CLS]', '[SEP]', '[PAD]'] else label_to_idx[i] for i in tags]
        # return {
        #     'sentence' : input, # str
        #     'input_ids' : torch.tensor(sentence['input_ids'], dtype=torch.long).to(devices),
        #     'token_type_id' : torch.tensor(sentence['token_type_ids'], dtype=torch.long).to(devices),
        #     'attention_mask' : torch.tensor(sentence['attention_mask'], dtype=torch.long).to(devices),
        #     'labels' : torch.tensor(convert_tags, dtype=torch.long).to(devices)
        # } # Collect_fn?

if __name__ == "__main__" :
    # load files
    df = load_files() # Have to control path (argparser)
    texts = df['form'].to_list()
    ne = df['NE'].to_list()

    # tokenizer
    tokenizer = get_tokenizer()
    train_texts, test_texts, train_ne, test_ne = train_test_split(texts, ne, test_size=0.2, random_state=42)
    test_dataset = CustomDataset(test_texts, test_ne)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle=True, collate_fn=collect_fn)
    cnt = 0
    for batch in test_loader :
        if cnt >= 1 :
            break
        cnt += 1
    # Return: texts: (Batch, token_num, 768) / labels : (Batch, token_num)