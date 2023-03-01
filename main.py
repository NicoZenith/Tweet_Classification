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
from transformers import DistilBertTokenizer
from transformers import BertForSequenceClassification, BertConfig, AdamW
# from transformers import DistilBertForSequenceClassification, AdamW, DistilBertConfig
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
from torch.nn import functional as F
import random 

print("b")

parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='dd', help='folder to output images and model checkpoints')
parser.add_argument('--lr', type=float, default= 2e-5, help='folder to output images and model checkpoints')
opt, unknown = parser.parse_known_args()
print(opt)

outf = opt.outf

for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

checkpoint_dir = 'checkpoints/' + outf
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

results_dir = os.path.join('', "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)



print("bi")

# Import Data
data_folder = 'data/'

test = pd.read_csv(data_folder + 'test.csv')
train = pd.read_csv(data_folder + 'train.csv')


# drop 'id' , 'keyword' and 'location' columns.
train.drop(columns=['id','keyword','location'], inplace=True)
test.drop(columns=['id','keyword','location'], inplace=True)

train["text"]=normalise_text(train["text"])
test["text"]=normalise_text(test["text"])

# split data into train and validation 
train_df, valid_df = train_test_split(train, test_size=0.20)

SEED = 52

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(train.head())


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # or any other tokenizer
train_dataset = CustomDataset(train_df, tokenizer)
val_dataset = CustomDataset(valid_df, tokenizer)

# Define the device to run the model on (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the modelreturns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model = model.to(device)

show_model_parameters(model)

# Set up optimizer 
optimizer = AdamW(model.parameters(),
                  lr = opt.lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


# Number of training epochs (authors recommend between 2 and 4)
epochs = 50
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


# for param in model.base_model.parameters():
#     param.requires_grad = False


# Set the seed value all over the place to make this reproducible.
seed_val = 52

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
show_every = 20

# Keep track of the best validation loss
best_val_loss = float('inf')


# For each epoch...
for epoch_i in range(0, epochs):
    
    store_train_loss = []
    store_train_acc = []
    store_val_loss = []
    store_val_acc = []
    
     # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()
    # Reset the total loss for this epoch.
    total_loss = 0
    
    model.train()
    
    
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        inputs_ids, attention_masks, labels = batch
        model.zero_grad()
        
        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(inputs_ids, 
                    attention_mask=attention_masks)
        
#         print(outputs[0])
#         print(labels.shape)
#         print(outputs[0].shape)
        loss = F.cross_entropy(outputs[0], labels)
#         print(outputs.shape)
        
        train_acc = compute_acc(outputs[0], labels)

        # Print the statistics
        store_train_loss.append(loss.item())
        store_train_acc.append(train_acc)
        
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        # Update the learning rate.
        scheduler.step()
        
        
        # Progress update every 40 batches.
        if step % show_every == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {} / {}.  Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            print('Training loss: %.3f  Training acc: %.3f'%(np.mean(store_train_loss[-show_every:]), np.mean(store_train_acc[-show_every:])) ) 
    
        
    # compute epoch loss and accuracy 
    train_losses.append(np.mean(store_train_loss))
    train_accuracies.append(np.mean(store_train_acc))
    
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    
    
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    
    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    
    # Evaluate data for one epoch
    for batch in val_dataloader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        val_inputs_ids, val_attention_masks, val_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():  
            val_outputs = model(val_inputs_ids,  
                            attention_mask=val_attention_masks)
            
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        val_logits = val_outputs[0]
        
        val_loss = F.cross_entropy(val_logits, val_labels)
        
        # Calculate the accuracy for this batch of test sentences.
        val_acc = compute_acc(val_logits, val_labels)
        
        store_val_loss.append(val_loss.item())
        store_val_acc.append(val_acc)
        
    
    # compute epoch loss and accuracy 
    mean_val_loss = np.mean(store_val_loss)
    val_losses.append(mean_val_loss)
    val_accuracies.append(np.mean(store_val_acc))
        
    # Report the final accuracy for this validation run.
    # Print loss and acc at the end of the epoch
    print("Epoch {}: Train Loss: {:.4f}, Validation Loss: {:.4f}, Train Accuracy: {:.2f}%, Validation Accuracy: {:.2f}%".format
    (epoch_i, train_losses[-1], val_losses[-1], train_accuracies[-1], val_accuracies[-1]))
    
     # Save figure 
    plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, results_dir + '/' + outf + "_loss_accuracies.pdf")

    # torch.save(model.state_dict(), os.path.join(checkpoint_dir, outf + "_trained.pth")) 

    # model.save_pretrained(checkpoint_dir + '/' + outf + '_weights')  

    # Check if validation loss has improved
    if np.mean(mean_val_loss < best_val_loss):
        best_val_loss = mean_val_loss
        # Save the model's state
        model.save_pretrained(checkpoint_dir)
        model.config.save_pretrained(checkpoint_dir)
        
print("")
print("Training complete!")
    
    