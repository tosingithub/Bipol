import torch
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
#import seaborn as sns
#import transformers
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset,  RandomSampler, SequentialSampler
import logging
from torch.optim import AdamW
import argparse
import pickle
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import utility as util
logging.basicConfig(level=logging.ERROR)
from transformers import RobertaTokenizer, RobertaForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification, DebertaTokenizer, DebertaForSequenceClassification, get_linear_schedule_with_warmup


jig_folder = '/home/shared_data/bipol/Jigsaw_kaggle/'
sbic_folder = '/home/shared_data/bipol/sbicv2/'
new_folder = '/home/shared_data/bipol/new/'

### If run from CLI, you may change the 2 default arguments below.
parser = argparse.ArgumentParser(description='Bias Detection')
parser.add_argument('--data_folder', type=str, default=jig_folder, help='location of the data')     # of sbic_folder
parser.add_argument('--model_name', type=str, default='roberta', help='name of the deep model')     # or deberta
parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=4, help='upper epoch limit')
parser.add_argument('--msave', type=str, default='newsaved_models', help='folder to save the finetuned model')
parser.add_argument('--batch_size', type=int, default=32, help='batch size') # smaller batch size for big model to fit GPU
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing the data
if args.data_folder == jig_folder:
    train_df = pd.read_csv(jig_folder + 'jig_train.csv', header=0)
    train_df = util.preprocess_pandas(train_df, list(train_df.columns))
    eval_df = pd.read_csv(jig_folder + 'jig_val.csv', header=0)
    eval_df = util.preprocess_pandas(eval_df, list(eval_df.columns))
    test_df = pd.read_csv(jig_folder + 'jig_test.csv', header=0)
    test_df = util.preprocess_pandas(test_df, list(test_df.columns))
elif args.data_folder == sbic_folder:
    train_df = pd.read_csv(sbic_folder + 'sbic_train.csv', header=0)
    train_df = util.preprocess_pandas(train_df, list(train_df.columns))
    eval_df = pd.read_csv(sbic_folder + 'sbic_val.csv', header=0)
    eval_df = util.preprocess_pandas(eval_df, list(eval_df.columns))
    test_df = pd.read_csv(sbic_folder + 'sbic_test.csv', header=0)
    test_df = util.preprocess_pandas(test_df, list(test_df.columns))
elif args.data_folder == new_folder:
    train_df = pd.read_csv(new_folder + 'new_train.csv', header=0)
    train_df = util.preprocess_pandas(train_df, list(train_df.columns))
    eval_df = pd.read_csv(new_folder + 'new_val.csv', header=0)
    eval_df = util.preprocess_pandas(eval_df, list(eval_df.columns))
    test_df = pd.read_csv(new_folder + 'new_test.csv', header=0)
    test_df = util.preprocess_pandas(test_df, list(test_df.columns))

def f1_score_func(labels, preds):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average=None), f1_score(labels_flat, preds_flat, average="weighted"), f1_score(labels_flat, preds_flat, average="macro")

def confusion_matrix_func(labels, preds):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    print(confusion_matrix(labels_flat, preds_flat))    # cell(r,c) 0,0- TN, 1,0- FN, 1,1- TP, 0,1- FP
    tn, fp, fn, tp = confusion_matrix(labels_flat, preds_flat).ravel()
    return tn, fp, fn, tp

def train(dataloader_train):
    print("Training...")
    loss_train_total = 0
    for batch in tqdm(dataloader_train): #, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False):
        optimizer.zero_grad()
        #model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2],}       
        outputs = model(**inputs)
        loss = outputs[0] # loss_fn(outputs, batch[2]) #
        #print("Loss ", loss)
        #print("Loss item ", loss.item())
        loss_train_total += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
            
    loss_train_avg = loss_train_total/len(dataloader_train)
    return loss_train_avg


def evaluate(dataloader_val):
    print("Evaluation...")
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    for batch in tqdm(dataloader_val):
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2],}

        with torch.no_grad():        
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            #logits = outputs[1]
            loss_val_total += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loss_fn = torch.nn.CrossEntropyLoss()
    if args.model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        model = RobertaForSequenceClassification.from_pretrained('roberta-base').to(device)
    elif args.model_name == 'deberta':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base', truncation=True, do_lower_case=True)
        model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base').to(device)
    elif args.model_name == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-generator', truncation=True, do_lower_case=True)
        model = ElectraForSequenceClassification.from_pretrained('google/electra-base-generator').to(device)

    traindata = train_df
    valdata = eval_df

    outfile =  args.model_name + '_' + args.data_folder.split('/')[-2] + '.txt'
    label_dict = {}         # For associating raw labels with indices/nos
    possible_labels = traindata.label.unique()
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    print(label_dict)       #
    traindata['label'] = traindata.label.replace(label_dict)                 # replace labels with their nos
    valdata['label'] = valdata.label.replace(label_dict)                 # replace labels with their nos
    print("Trainset distribution: \n", traindata['label'].value_counts())                           # check data distribution

    encoded_data_train = tokenizer.batch_encode_plus(traindata.comment_text.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
    encoded_data_val = tokenizer.batch_encode_plus(valdata.comment_text.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
    labels_train = torch.tensor(traindata['label'].values)
    labels_val = torch.tensor(valdata['label'].values)    
    input_ids_train = encoded_data_train['input_ids'].to(device)
    attention_masks_train = encoded_data_train['attention_mask'].to(device)
    input_ids_val = encoded_data_val['input_ids'].to(device)
    attention_masks_val = encoded_data_val['attention_mask'].to(device)
    labels_train = labels_train.to(device)
    labels_val = labels_val.to(device)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=args.batch_size)
    dataloader_validation = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*args.epochs)

    best_loss = None
    best_model = None
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(dataloader_train)
        val_loss, predictions, true_vals = evaluate(dataloader_validation)
        val_f1, val_f1_w, val_f1_mac = f1_score_func(true_vals, predictions)
        epoch_time_elapsed = time.time() - epoch_start_time
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} '.format(epoch, train_loss, val_loss) + f'F1: {val_f1}, weighted F1: {val_f1_w}, macro F1: {val_f1_mac}')       
        with open(outfile, "a+") as f:
            s = f.write('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} '.format(epoch, train_loss, val_loss) + f'F1: {val_f1}, weighted F1: {val_f1_w}, macro F1: {val_f1_mac}' + "\n")
        #if not best_loss or val_f1_w < best_loss:
        if not best_loss or val_loss < best_loss:
            #best_model = model                 # Not needed here
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.msave)  # transformers save
            tokenizer.save_pretrained(args.msave)
            best_loss = val_loss
