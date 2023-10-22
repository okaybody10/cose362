import torch
import torch.nn as nn
from kobert_transformers import get_tokenizer, get_kobert_model
from TorchCRF import CRF
from svm import KernelSVM
from process import tagging
from torch.nn.utils.rnn import pad_sequence

# Input: (sentence, ne_lists) <- CustomDataset
# DataLoader: (Batch, Sentence, ne_lists)
def label_init() :
    label_list = ['PS', 'FD', 'TR', 'AF', 'OG', 'LC', 'CV', 'DT', 'TI', 'TI', 'QT', 'EV', 'AM', 'PT', 'MT', "TM"] 
    label_fin = []
    label_fin += ['B-' + i for i in label_list]
    label_fin += ['O']
    label_fin += ['I-' + i for i in label_list]
    label_fin += ['[CLS]']
    label_fin += ['[SEP]']
    label_fin += ['[PAD]']
    label_to_idx = {label: idx for idx, label in enumerate(label_fin)}
    idx_to_label = {idx: label for idx, label in enumerate(label_fin)}
    return label_fin, label_to_idx, idx_to_label

class NERBertCRF(nn.Module) :
    def __init__(self, num_classes= 36, dropout = 0.1) -> None: 
        super(NERBertCRF, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes + 1
        self.tokenizer = get_tokenizer()
        self.bert = get_kobert_model()
        self.dropout = nn.Dropout(p=dropout) # Setting dropout
        self.linear = nn.Linear(768, self.num_classes)
        self.CRF = CRF(self.num_classes) # Last CRF Layers

        self.label_lists, self.label_to_idx, self.idx_to_label = label_init()
    
    def forward(self, texts, ne) :
        # texts input: Only texts
        # label: Tag results, automatically padding
        
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors= 'pt').to(self.device) # Return input_token, attention_mask, input_ids with padding
        tags = [tagging(self.tokenizer.convert_ids_to_tokens(index), label, bert_check=True) for (index, label) in zip(tokens['input_ids'], ne)]
        # Convert tags
        labels = torch.LongTensor([[self.label_to_idx[i] for i in tag] for tag in tags]).to(self.device) # For calculation of cross entropy
        masks = torch.BoolTensor([[0 if i in ['[PAD]'] else 1 for i in tag] for tag in tags]).to(self.device)
        sentence_output = self.bert(**tokens)[0]
        outpus_d = self.dropout(sentence_output)
        emission = self.linear(outpus_d)
        log_likelihood, predict_tags = self.CRF.forward(emission, labels, masks), self.CRF.viterbi_decode(emission, masks)
        PAD_ID = self.label_to_idx['[PAD]']
        outputs = [torch.LongTensor(i) for i in predict_tags]
        outputs.append(labels[-1])
        outputs = torch.LongTensor(pad_sequence(outputs, batch_first=True, padding_value=PAD_ID)[:-1]).to(self.device)
        return log_likelihood, predict_tags, labels, outputs

class NERBertSVM(nn.Module) :
    def __init__(self, num_classes= 33, dropout = 0.1) -> None: 
        super(NERBertSVM, self).__init__()

        self.tokenizer = get_tokenizer()
        self.bert = get_kobert_model()
        self.dropout = nn.Dropout(p=dropout) # Setting dropout
        self.linear = nn.Linear((768, num_classes))

        self.label_lists, self.label_to_idx, self.idx_to_label = label_init()
    
    def forward(self, texts, ne) :
        # texts input: Only texts
        # label: Tag results, automatically padding
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors= 'pt') # Return input_token, attention_mask, input_ids with padding
        tags = [tagging(self.tokenizer.convert_ids_to_tokens(index), label) for (index, label) in zip(tokens['input_ids'], ne)]
        # Convert tags
        labels = [torch.tensor([-100 if i in ['[CLS]', '[SEP]', '[PAD]'] else self.label_to_idx[i] for i in tag]) for tag in tags] # For calculation of cross entropy
        masks = [torch.tensor([0 if i in ['[CLS]', '[SEP]', '[PAD]'] else 1 for i in tag]) for tag in tags]
        sentence_output = self.bert(**tokens)[0]
        outpus_d = self.dropout(sentence_output)
        ## TODO: Edit
        # emission = self.linear(outpus_d)
        # Predict loss
        # log_likelihood, predict_tags = self.CRF(emission, labels, masks), self.CRF.viterbi_decode(emission, masks)
        pass
        # return log_likelihood, predict_tags
