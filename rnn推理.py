import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
import re
from operator import itemgetter
from tqdm import tqdm
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parameter = {'embedding_dim':256,
         'kernel_size':64,
        'epoch':30,
         'dropout':0.1,
         'lr':0.1,
         'n_layers':2,
         'hid_dim':128,
        'batch_size':100}


def built_dataset(parameter):
    word_vocab = defaultdict(int)
    word_vocab['pad'] = 0
    label_vocab = defaultdict(int)
    chars = []
    labels = []
    datas = pd.read_csv('classification_data.csv',sep = ',')
    for data in datas.text:
        for i in data:
            word_vocab[i] +=1
        chars.append(list(re.sub(r'[ \n\t]','',data)))

    for label in datas.label:  
        labels.append(label)         
        label_vocab[label] +=1
    word2id = dict(zip(word_vocab.keys(),range(len(word_vocab.keys()))))
    id2word = dict(zip(range(len(word_vocab.keys())),word_vocab.keys()))
    label2id = dict(zip(label_vocab.keys(),range(len(label_vocab.keys()))))    
    id2label = dict(zip(range(len(label_vocab.keys())),label_vocab.keys()))
    parameter['word2id'] = word2id
    parameter['word_size'] = len(word_vocab.keys())
    parameter['id2word'] = id2word
    parameter['label2id'] = label2id
    parameter['id2label'] = id2label
    parameter['output_size'] = len(label2id)
    return np.array(chars), np.array(labels)



def batch_yield(parameter,chars,labels,shuffle = True):
    if shuffle:
        permutation = np.random.permutation(len(chars))
        chars = chars[permutation]
        labels = labels[permutation]
        batch_chars_ids = []
        batch_labels = []
        len_x = []
    for iters in tqdm(range(len(chars))):             
        len_x.append(len(chars[iters]))
        batch_labels.append(labels[iters]) 
        try:
            batch_chars_ids.append(list(itemgetter(*chars[iters])(parameter['word2id'])))
        except:
            batch_chars_ids.append([itemgetter(*chars[iters])(parameter['word2id']),0])
        
        if len(batch_chars_ids)>=parameter['batch_size']:
            batch_chars_ids = [batch_chars_id+[0]*(max(len_x)-len(batch_chars_id)) \
                               for batch_chars_id in batch_chars_ids]   
            yield torch.from_numpy(np.array(batch_chars_ids)).to(device),\
            torch.from_numpy(np.array(batch_labels)).to(device).long(),None,None
            batch_chars_ids = []
            batch_labels = []
            len_x = [] 
    batch_chars_ids = [batch_chars_id+[0]*(max(len_x)-len(batch_chars_id)) \
                               for batch_chars_id in batch_chars_ids]           
    yield torch.from_numpy(np.array(batch_chars_ids)).to(device),\
            torch.from_numpy(np.array(batch_labels)).to(device).long(),True,epoch   
   
    

import torch.nn.functional as F
from torch import nn,optim

class bilstm(nn.Module):
    def __init__(self,parameter):
        super().__init__()  
        word_size = parameter['word_size']
        embedding_dim = parameter['embedding_dim']
        self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx = 0)
        hidden_dim = parameter['hid_dim']
        num_layers = parameter['n_layers']
        dropout = parameter['dropout']
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional = True,\
                        batch_first = True, dropout = dropout)
        output_size = parameter['output_size']
        self.fc = nn.Linear(hidden_dim*2,output_size)
        #self.crf = CRF(output_size,batch_first = True)
     
    def forward(self,x):
        out = self.embedding(x)
        out,(h,c) = self.lstm(out)
        out = out[:,-1]
        out = self.fc(out)
        return out
    
chars,labels = built_dataset(parameter)
train_chars,test_chars,train_labels,test_labels = train_test_split(chars,labels, test_size=0.2, random_state=42)

model = bilstm(parameter).to(device)
model.load_state_dict(torch.load('model-rnn.h5'))
model.eval()
#optimizer = torch.optim.SGD(model.parameters(),lr=0.001, momentum=0.95, nesterov=True)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.7)
#criterion = nn.CrossEntropyLoss()   
train_yield = batch_yield(parameter,test_chars,test_labels)
count_dict = {}
min_loss = float('inf')
while 1:
    seqs,label,keys,epoch = next(train_yield)

    out = model(seqs)
    #loss = criterion(out,label.long().to(device))
    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()
    #loss_cal.append(loss.item())
    #if epoch is not None:
        #if (epoch+1)%1 == 0:
            #loss_ = sum(loss_cal)/len(loss_cal)
            #if loss_ < min_loss:
                #min_loss = loss_
                #torch.save(model.state_dict(), 'model-cnn.h5')
            #print('epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, \
                            #parameter['epoch'],loss_))
        #loss_cal = []
        #scheduler.step()
    pred,pred_index = torch.max(F.softmax(out,-1),-1)
    pred_index = pred_index.cpu().numpy()
    label = label.cpu().numpy()
    right = (pred_index==label)
    accs = []
    acc = sum(right)/len(right)
    accs.append(acc)
    for i in range(0,parameter['output_size']):
        if i not in count_dict:
            count_dict[i]={'pred_pos':len(pred_index[pred_index == i]),
                           'real_pos':len(label[label==i]),
                           'true_pos':len(label[(right & (pred_index == i))])}
        else:
            count_dict[i]['pred_pos']+=len(pred_index[pred_index == i])
            count_dict[i]['real_pos']+=len(label[label==i])
            count_dict[i]['true_pos']+=len(label[(right & (pred_index == i))])
    if keys:
        break        

#a = list(range(0,parameter['output_size']))
#label_name = list(itemgetter(*a)(parameter['id2label']))
count_pandas = pd.DataFrame(count_dict.values(), columns=['pred_pos','real_pos','true_pos'])
count_pandas['pre'] = count_pandas['true_pos']/count_pandas['pred_pos']
count_pandas['recall'] = count_pandas['true_pos']/count_pandas['real_pos']
count_pandas['f1'] = 2*count_pandas['pre']*count_pandas['recall']/(count_pandas['pre']+count_pandas['recall'])
pre_mean = count_pandas['pre'].mean()
recall_mean = count_pandas['recall'].mean()
f1_mean = count_pandas['f1'].mean()
count_pandas.loc['mean'] = ['/','/','/',pre_mean,recall_mean,f1_mean]
#label_name = itemgetter(*a)(parameter['id2label'])

print(count_pandas)  
print(sum(accs)/len(accs))