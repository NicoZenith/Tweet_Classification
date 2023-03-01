# Check the files in Data
# Check the files in Data
import os
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from torch.utils.data import Dataset, DataLoader
import torch
from utils import * 
from transformers import BertTokenizer # assuming you want to use BERT tokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
from torch.nn import functional as F
import random 

print("b")

parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='dd', help='folder to output images and model checkpoints')
opt, unknown = parser.parse_known_args()
print(opt)

outf = opt.outf
checkpoint_dir = 'checkpoints/'+outf 

data_folder = 'data/'

test = pd.read_csv(data_folder + 'test.csv')

submission = pd.read_csv(data_folder +  'sample_submission.csv')
submission['target'] = 0

test.drop(columns=['id','keyword','location'], inplace=True)

test["text"]=normalise_text(test["text"])

# split data into train and validation 
# train_df, valid_df = train_test_split(test)

SEED = 52

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# print(train.head())


test['target'] = 0

print(test.head())

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # or any other tokenizer
test_dataset = CustomDataset(test, tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(checkpoint_dir)
# Tell pytorch to run this model on the GPU.
model = model.to(device)

show_model_parameters(model)


# Load the state dictionary into the model

# Set the model to evaluation mode
model.eval()


store_test_acc = []

# For each batch of training data...
for step, batch in enumerate(test_dataloader):
    
    
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    inputs_ids, attention_masks, labels = batch
    with torch.no_grad():
        outputs = model(inputs_ids, 
                    token_type_ids=None, 
                    attention_mask=attention_masks)
        predictions = torch.max(outputs[0], dim=1)[1]
        submission.loc[step*len(predictions): (step+1)*len(predictions) - 1, 'target' ] = predictions.cpu().numpy()
        submission.to_csv(data_folder+'sample_submission.csv', index=False)
        # store_test_acc.append(compute_acc(outputs[0], labels))

# print("Training accuracy: {:.2f}%".format(np.mean(store_test_acc)))

print("Finished testing")