import logging
from statistics import mean
import argparse
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve
import wandb
#from simpletransformers.classification import ClassificationArgs, ClassificationModel
import utility
import re
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification, DebertaTokenizer, DebertaForSequenceClassification
from tqdm.auto import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import copy
import argparse
import os
import time
from datasets import load_dataset, list_datasets
from transformers import AutoModel,AutoTokenizer

mab_swedish = '/home/shared_data/bipol/mab_swedish/'

### --modeldir REMEMBER to match with --model_name
rob_new = '/home/oluade/e_bipol/saved_models2/roberta_new/'
rob_sbic = '/home/oluade/e_bipol/saved_models2/roberta_sbic/'
rob_jig = '/home/oluade/e_bipol/saved_models2/roberta_jig/'
elec_sbic = '/home/oluade/e_bipol/saved_models2/electra_sbic/'
elec_new = '/home/oluade/e_bipol/saved_models2/electra_new/'

elec_jig = '/home/oluade/e_bipol/saved_models2/electra_jig/'
deb_new = '/home/oluade/e_bipol/saved_models2/deberta_new/'
deb_sbic = '/home/oluade/e_bipol/saved_models2/deberta_sbic/'
deb_jig = '/home/oluade/e_bipol/saved_models2/deberta_jig/'


parser = argparse.ArgumentParser(description='IJCNN - Bias Detection')
#Start data with - copa, cb, axg, wsc, rte, boolq, multirc, record
#parser.add_argument('--data', type=str, default='boolq', help='location of the data')     # boolq, cb, copa, axb, axg
parser.add_argument('--data_folder', type=str, default='copa', help='location of the data')     # of sbic_folder
parser.add_argument('--model_name', type=str, default='electra', help='name of the deep model')     # or deberta
parser.add_argument('--axes_folder', type=str, default='axes/swedish/', help='name of the folder containing the bias axes files')     # or deberta
parser.add_argument('--modeldir', type=str, default=elec_new, help='directory of the model checkpoint')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def f1_score_func(labels, preds_flat):
    return f1_score(labels, preds_flat, average=None), f1_score(labels, preds_flat, average="weighted"), f1_score(labels, preds_flat, average="macro")

def confusion_matrix_func(labels, preds_flat):
    print(confusion_matrix(labels, preds_flat))  # cell(r,c) 0,0- TN, 1,0- FN, 1,1- TP, 0,1- FP
    tn, fp, fn, tp = confusion_matrix(labels, preds_flat).ravel()
    return tn, fp, fn, tp

def prec_rec(labels):
    precision_recall_curve()



if __name__=="__main__":

    if args.data_folder == 'boolq':
        data_train = load_dataset('super_glue', 'boolq', split='train')      # 9427 entries
        #data_val = load_dataset('super_glue', 'boolq', split='validation')        # 3270 entries
        test_df_ = pd.DataFrame(data_train['passage'], columns =['comment_text'])
        test_df = test_df_.drop_duplicates(ignore_index=True)                                         #
        test_df = utility.preprocess_pandas(test_df, list(test_df.columns))
        test_df.insert(len(test_df.columns), 'id', range(1, 1 + len(test_df)))

    elif args.data_folder == 'cb':
        #data_train = load_dataset('super_glue', 'cb')
        #print(data_train)
        data_train = load_dataset('super_glue', 'cb', split='train')      # 250 entries
        #data_val = load_dataset('super_glue', 'cb', split='validation')        # 56 entries
        #data_test = load_dataset('super_glue', 'copa', split='test')        # 250 entries
        test_df_ = pd.DataFrame(data_train['premise'], columns =['comment_text'])
        test_df = test_df_.drop_duplicates(ignore_index=True)                                         #
        test_df = utility.preprocess_pandas(test_df, list(test_df.columns))
        test_df.insert(len(test_df.columns), 'id', range(1, 1 + len(test_df)))

    elif args.data_folder == 'copa':
        data_train = load_dataset('super_glue', 'copa', split='train')      # 400 entries
        #data_val = load_dataset('super_glue', 'copa', split='validation')        # 100 entries
        #data_test = load_dataset('super_glue', 'copa', split='test')        # 500 entries
        test_df_ = pd.DataFrame(data_train['premise'], columns =['comment_text'])
        test_df = test_df_.drop_duplicates(ignore_index=True)                                         #
        test_df = utility.preprocess_pandas(test_df, list(test_df.columns))
        test_df.insert(len(test_df.columns), 'id', range(1, 1 + len(test_df)))

    elif args.data_folder == 'axg':        # Designed for gender balance, it seems
        data_test = load_dataset('super_glue', 'axg', split='test')      # 356 entries
        test_df_ = pd.DataFrame(data_test['premise'], columns =['comment_text'])
        test_df = test_df_.drop_duplicates(ignore_index=True)                                         #
        test_df = utility.preprocess_pandas(test_df, list(test_df.columns))
        test_df.insert(len(test_df.columns), 'id', range(1, 1 + len(test_df)))

    elif args.data_folder == 'multirc':
        data_train = load_dataset('super_glue', 'multirc', split='train')      # 27243 entries
        #data_val = load_dataset('super_glue', 'multirc', split='validation')        # 4848 entries
        #data_val = load_dataset('super_glue', 'multirc', split='test')        # 9693 entries
        test_df_ = pd.DataFrame(data_train['paragraph'], columns =['comment_text'])
        test_df = test_df_.drop_duplicates(ignore_index=True)                                         #
        test_df = utility.preprocess_pandas(test_df, list(test_df.columns))
        test_df.insert(len(test_df.columns), 'id', range(1, 1 + len(test_df)))

    elif args.data_folder == 'record':
        data_train = load_dataset('super_glue', 'record', split='train')      # 100730 entries
        #data_val = load_dataset('super_glue', 'record', split='validation')        # 10000 entries
        #data_test = load_dataset('super_glue', 'record', split='test')        # 10000 entries
        test_df_ = pd.DataFrame(data_train['passage'], columns =['comment_text'])
        test_df = test_df_.drop_duplicates(ignore_index=True)                                         #
        test_df = utility.preprocess_pandas(test_df, list(test_df.columns))
        test_df.insert(len(test_df.columns), 'id', range(1, 1 + len(test_df)))

    elif args.data_folder == 'rte':
        data_train = load_dataset('super_glue', 'rte', split='train')      # 2490 entries
        #data_val = load_dataset('super_glue', 'rte', split='validation')        # 277 entries
        #data_test = load_dataset('super_glue', 'rte', split='test')        # 3000 entries
        test_df_ = pd.DataFrame(data_train['premise'], columns =['comment_text'])
        test_df = test_df_.drop_duplicates(ignore_index=True)                                         #
        test_df = utility.preprocess_pandas(test_df, list(test_df.columns))
        test_df.insert(len(test_df.columns), 'id', range(1, 1 + len(test_df)))

    elif args.data_folder == 'wsc':
        data_train = load_dataset('super_glue', 'wsc', split='train')      # 554 entries
        #data_val = load_dataset('super_glue', 'wsc', split='validation')        # 104 entries
        #data_val = load_dataset('super_glue', 'wsc', split='test')        # 146 entries
        test_df_ = pd.DataFrame(data_train['text'], columns =['comment_text'])
        test_df = test_df_.drop_duplicates(ignore_index=True)                                         #
        test_df = utility.preprocess_pandas(test_df, list(test_df.columns))
        test_df.insert(len(test_df.columns), 'id', range(1, 1 + len(test_df)))

    elif args.data_folder == mab_swedish:
        train_df = pd.read_csv(mab_swedish + 'swedish_mab_train.csv', header=0)
        train_df = utility.preprocess_pandas(train_df, list(train_df.columns))
        eval_df = pd.read_csv(mab_swedish + 'swedish_mab_val.csv', header=0)
        eval_df = utility.preprocess_pandas(eval_df, list(eval_df.columns))
        test_df = pd.read_csv(mab_swedish + 'swedish_mab_test.csv', header=0)
        test_df = utility.preprocess_pandas(test_df, list(test_df.columns))

    # eval_test()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load lexicon from external files to build a dictionary of axes
    axes, ax_s, cnt = {}, [], 0
    filenames = sorted(os.listdir(args.axes_folder))                        # list of filenames sorted
    for file in filenames:
        # fill axes in the main dict with file names
        file = file.lower()
        axis_name = file.split('_')[0]
        if axis_name not in axes:
            axes[axis_name] = []
            ax_s = []                                                 # re-initialize
        if file.endswith(".txt"):
            file_path = args.axes_folder + file
            with open(file_path, encoding='utf8') as f:
                ax_shade = {}
                for line in f:
                    linestr = ' ' + line.strip() + ' '                                # pre & suffix spaces to get lone words
                    linestr = linestr.lower()
                    if linestr not in ax_shade:
                        ax_shade[linestr] = 0
            ax_s.append(ax_shade)
        axes[axis_name] = ax_s

    test_df_ = None                                                             # initialize count of biased samples to nothing for use in corpus-level bipol
    axes2 = copy.deepcopy(axes)                                                 # for explainability
    predictions, tvals = [], []
    bipol_sent, bip_sent, bipol_sent_agg, bipol_cop = 0, 0, 0, 0                # initialize important variables
    eval_start_time = time.time()
    ### Code refactoring should be done below in the future
    if args.model_name == 'roberta':
        # Evaluate test set
        tokenizer = RobertaTokenizer.from_pretrained(args.modeldir, truncation=True, do_lower_case=True)
        model = RobertaForSequenceClassification.from_pretrained(args.modeldir)

        # Corpus-level & sentence-level evaluation:
        print("No of classes: ", model.num_labels)
        # test_df = test_df[:8]          # small for trials
        print('Total no: ', len(test_df))

        test_df2 = test_df.copy()                                           # test_df2 for label replacement for roberta below
        print(test_df.head())
        if not 'prediction' in test_df.columns:
            test_df.insert(len(test_df.columns), 'prediction', '')
        
        # label_dict = {}                                                     # For associating raw labels with indices/nos
        # if args.data_folder != 'squad' and args.data_folder != 'imdb':
        #     possible_labels = train_df.label.unique()
        #     for index, possible_label in enumerate(possible_labels):
        #         label_dict[possible_label] = index
        #     print(label_dict)                                                   # {'unbiased': 0, 'biased': 1}
        for rno in range(len(test_df['id'])):
            input_ids = tokenizer(test_df['comment_text'][rno], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').input_ids
            outputs = model(input_ids)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)        # store the predictions in a list
        predictions = np.concatenate(predictions, axis=0)
        preds_flat = np.argmax(predictions, axis=1).flatten()
        # Using list comprehension below should be faster
        for rno in range(len(test_df['id'])):
            test_df.loc[(test_df['id'] == test_df['id'][rno]) & (preds_flat[rno] == 1), 'prediction'] = 'biased'
            test_df.loc[(test_df['id'] == test_df['id'][rno]) & (preds_flat[rno] == 0), 'prediction'] = 'unbiased'
            # sentence-level bipol for biased samples only - iterate through the dict of axes and calculate
            if preds_flat[rno] == 1:
                #print(" Prediction is 1 - BIASED", rno)
                for a_k, a_v in axes.items():
                    list_shade = []                                                 # initialize
                    for b in range(len(a_v)):                                       # Go to next dict in list
                        inshade_cnt = 0
                        for k in a_v[b].keys():
                            a_v[b][k] = test_df['comment_text'][rno].count(k)       # In future make the column name dynamic
                            axes2[a_k][b][k] = axes2[a_k][b][k] + a_v[b][k]         # sum all sensitive tokens for explainability
                            #print(a_v[b][k])
                            inshade_cnt += a_v[b][k]
                        #print(inshade_cnt)
                        list_shade.append(inshade_cnt)
                    #print('Shades ', list_shade)
                    bip_divis = sum(list_shade)
                    if len(list_shade) < 1:
                        bip_sent1 = 0
                    else:
                        bip_sent1 = max(list_shade)
                        list_shade.remove(bip_sent1)
                    if len(list_shade) < 1:                                         # check empty cases
                        bip_sent2 = 0
                    else:
                        bip_sent2 = max(list_shade)                                 # get max after the 1st max
                    if bip_divis > 0:                                           # To avoid divide by zero error
                        bip_sent = abs(bip_sent1 - bip_sent2) / bip_divis
                    bipol_sent += bip_sent                                       # sum over axes
                
                #print('Total over axis ', bipol_sent, len(axes))
                bipol_sent = bipol_sent/len(axes)                                    # average over axes per sentence
                #print('After division - Total over axis ', bipol_sent)
                bipol_sent_agg += bipol_sent                                        # sum over multiple sentences
                #print('Total over sentences ', bipol_sent_agg)
                bipol_sent = 0                                                      # re-initialize for next sentence
        
        test_df_ = test_df[test_df['prediction'] == 'biased']                       # Find out the total of biased samples
        print(test_df_.head())
        print('biased total no: ', len(test_df_))
        if len(test_df_) > 0:
            bipol_sent_agg = bipol_sent_agg/len(test_df_)                           # Overall sentence-level bipol
            print('After division - Total over sentences ', bipol_sent_agg)
        else:
            bipol_sent_agg = 0                                                      # return zero for <= 0 (hopefully shouldn't be)

        if 'label' in test_df2.columns:                                             # In future, make this column dynamic
            test_df2['label'] = test_df2.label.replace(label_dict)                  # replace labels with their nos for F1 scores
            tvals = test_df2['label'].values.tolist()

    elif args.model_name == 'deberta':
        tokenizer = DebertaTokenizer.from_pretrained(args.modeldir, truncation=True, do_lower_case=True)
        model = DebertaForSequenceClassification.from_pretrained(args.modeldir)

        # Corpus-level & sentence-level evaluation:
        print("No of classes: ", model.num_labels)
        #test_df = test_df[:8]          # small for trials
        print('Total no: ', len(test_df))

        test_df2 = test_df.copy()                                           # test_df2 for label replacement for roberta below
        print(test_df.head())
        if not 'prediction' in test_df.columns:
            test_df.insert(len(test_df.columns), 'prediction', '')
        
        label_dict = {}                                                     # For associating raw labels with indices/nos
        # possible_labels = train_df.label.unique()
        # for index, possible_label in enumerate(possible_labels):
        #     label_dict[possible_label] = index
        # print(label_dict)                                                   # {'unbiased': 0, 'biased': 1}
        for rno in range(len(test_df['id'])):
            input_ids = tokenizer(test_df['comment_text'][rno], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').input_ids
            outputs = model(input_ids)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)        # store the predictions in a list
        predictions = np.concatenate(predictions, axis=0)
        preds_flat = np.argmax(predictions, axis=1).flatten()
        # Using list comprehension below should be faster
        for rno in range(len(test_df['id'])):
            test_df.loc[(test_df['id'] == test_df['id'][rno]) & (preds_flat[rno] == 1), 'prediction'] = 'biased'
            test_df.loc[(test_df['id'] == test_df['id'][rno]) & (preds_flat[rno] == 0), 'prediction'] = 'unbiased'
            # sentence-level bipol for biased samples only - iterate through the dict of axes and calculate
            if preds_flat[rno] == 1:
                #print(" Prediction is 1 - BIASED", rno)
                for a_k, a_v in axes.items():
                    list_shade = []                                                 # initialize
                    for b in range(len(a_v)):                                       # Go to next dict in list
                        inshade_cnt = 0
                        for k in a_v[b].keys():
                            a_v[b][k] = test_df['comment_text'][rno].count(k)       # In future make the column name dynamic
                            axes2[a_k][b][k] = axes2[a_k][b][k] + a_v[b][k]         # sum all sensitive tokens for explainability
                            #print(a_v[b][k])
                            inshade_cnt += a_v[b][k]
                        #print(inshade_cnt)
                        list_shade.append(inshade_cnt)
                    #print('Shades ', list_shade)
                    bip_divis = sum(list_shade)
                    if len(list_shade) < 1:
                        bip_sent1 = 0
                    else:
                        bip_sent1 = max(list_shade)
                        list_shade.remove(bip_sent1)
                    if len(list_shade) < 1:                                         # check empty cases
                        bip_sent2 = 0
                    else:
                        bip_sent2 = max(list_shade)                                 # get max after the 1st max
                    if bip_divis > 0:                                           # To avoid divide by zero error
                        bip_sent = abs(bip_sent1 - bip_sent2) / bip_divis
                    bipol_sent += bip_sent                                       # sum over axes
                
                #print('Total over axis ', bipol_sent, len(axes))
                bipol_sent = bipol_sent/len(axes)                                    # average over axes per sentence
                #print('After division - Total over axis ', bipol_sent)
                bipol_sent_agg += bipol_sent                                        # sum over multiple sentences
                #print('Total over sentences ', bipol_sent_agg)
                bipol_sent = 0                                                      # re-initialize for next sentence
        
        test_df_ = test_df[test_df['prediction'] == 'biased']                       # Find out the total of biased samples
        print(test_df_.head())
        print('biased total no: ', len(test_df_))
        if len(test_df_) > 0:
            bipol_sent_agg = bipol_sent_agg/len(test_df_)                           # Overall sentence-level bipol
            print('After division - Total over sentences ', bipol_sent_agg)
        else:
            bipol_sent_agg = 0                                                      # return zero for <= 0 (hopefully shouldn't be)

        if 'label' in test_df2.columns:                                             # In future, make this column dynamic
            test_df2['label'] = test_df2.label.replace(label_dict)                  # replace labels with their nos for F1 scores
            tvals = test_df2['label'].values.tolist()

    elif args.model_name == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained(args.modeldir, truncation=True, do_lower_case=True)
        model = ElectraForSequenceClassification.from_pretrained(args.modeldir)

        # Corpus-level & sentence-level evaluation:
        print("No of classes: ", model.num_labels)
        #test_df = test_df[:8]          # small for trials
        print('Total no: ', len(test_df))

        test_df2 = test_df.copy()                                           # test_df2 for label replacement for roberta below
        print(test_df.head())
        if not 'prediction' in test_df.columns:
            test_df.insert(len(test_df.columns), 'prediction', '')
        
        label_dict = {}                                                     # For associating raw labels with indices/nos
        # possible_labels = train_df.label.unique()
        # for index, possible_label in enumerate(possible_labels):
        #     label_dict[possible_label] = index
        # print(label_dict)                                                   # {'unbiased': 0, 'biased': 1}
        for rno in range(len(test_df['id'])):
            input_ids = tokenizer(test_df['comment_text'][rno], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').input_ids
            outputs = model(input_ids)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)        # store the predictions in a list
        predictions = np.concatenate(predictions, axis=0)
        preds_flat = np.argmax(predictions, axis=1).flatten()
        # Using list comprehension below should be faster
        for rno in range(len(test_df['id'])):
            test_df.loc[(test_df['id'] == test_df['id'][rno]) & (preds_flat[rno] == 1), 'prediction'] = 'biased'
            test_df.loc[(test_df['id'] == test_df['id'][rno]) & (preds_flat[rno] == 0), 'prediction'] = 'unbiased'
            # sentence-level bipol for biased samples only - iterate through the dict of axes and calculate
            if preds_flat[rno] == 1:
                #print(" Prediction is 1 - BIASED", rno)
                for a_k, a_v in axes.items():
                    list_shade = []                                                 # initialize
                    for b in range(len(a_v)):                                       # Go to next dict in list
                        inshade_cnt = 0
                        for k in a_v[b].keys():
                            a_v[b][k] = test_df['comment_text'][rno].count(k)       # In future make the column name dynamic
                            axes2[a_k][b][k] = axes2[a_k][b][k] + a_v[b][k]         # sum all sensitive tokens for explainability
                            #print(a_v[b][k])
                            inshade_cnt += a_v[b][k]
                        #print(inshade_cnt)
                        list_shade.append(inshade_cnt)
                    #print('Shades ', list_shade)
                    bip_divis = sum(list_shade)
                    if len(list_shade) < 1:
                        bip_sent1 = 0
                    else:
                        bip_sent1 = max(list_shade)
                        list_shade.remove(bip_sent1)
                    if len(list_shade) < 1:                                         # check empty cases
                        bip_sent2 = 0
                    else:
                        bip_sent2 = max(list_shade)                                 # get max after the 1st max
                    if bip_divis > 0:                                           # To avoid divide by zero error
                        bip_sent = abs(bip_sent1 - bip_sent2) / bip_divis
                    bipol_sent += bip_sent                                       # sum over axes
                
                #print('Total over axis ', bipol_sent, len(axes))
                bipol_sent = bipol_sent/len(axes)                                    # average over axes per sentence
                #print('After division - Total over axis ', bipol_sent)
                bipol_sent_agg += bipol_sent                                        # sum over multiple sentences
                #print('Total over sentences ', bipol_sent_agg)
                bipol_sent = 0                                                      # re-initialize for next sentence
        
        test_df_ = test_df[test_df['prediction'] == 'biased']                       # Find out the total of biased samples
        print(test_df_.head())
        print('biased total no: ', len(test_df_))
        if len(test_df_) > 0:
            bipol_sent_agg = bipol_sent_agg/len(test_df_)                           # Overall sentence-level bipol
            print('After division - Total over sentences ', bipol_sent_agg)
        else:
            bipol_sent_agg = 0                                                      # return zero for <= 0 (hopefully shouldn't be)

        if 'label' in test_df2.columns:                                             # In future, make this column dynamic
            test_df2['label'] = test_df2.label.replace(label_dict)                  # replace labels with their nos for F1 scores
            tvals = test_df2['label'].values.tolist()

    elif args.model_name == 'sv_bert':
        #tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased', truncation=True, do_lower_case=True) # , use_fast=False
        #model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')
        #model = T5Model("mt5", "google/mt5-small", args=model_args, use_cuda=True) # sweep_config=wandb.sconfig,
        model = T5ForConditionalGeneration.from_pretrained("mt5", "google/mt5-small")
        tokenizer = T5Tokenizer.from_pretrained("mt5", "google/mt5-small")

        # Corpus-level & sentence-level evaluation:
        #print("No of classes: ", model.num_labels)
        test_df = test_df[:10]          # small for trials
        print('Total no: ', len(test_df))

        test_df2 = test_df.copy()                                           # test_df2 for label replacement for roberta below
        print(test_df.head())
        if not 'prediction' in test_df.columns:
            test_df.insert(len(test_df.columns), 'prediction', '')
        
        label_dict = {}                                                     # For associating raw labels with indices/nos
        # possible_labels = train_df.label.unique()
        # for index, possible_label in enumerate(possible_labels):
        #     label_dict[possible_label] = index
        # print(label_dict)                                                   # {'unbiased': 0, 'biased': 1}
        for rno in range(len(test_df['id'])):
            input_ids = tokenizer(test_df['comment_text'][rno], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').input_ids
            outputs = model(input_ids)
            logits = outputs[0]
            #preds = torch.argmax(logits, dim=1).flatten()
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)        # store the predictions in a list
        print(predictions)
        predictions = np.concatenate(predictions, axis=0)
        preds_flat = np.argmax(predictions, axis=1).flatten()
        # Using list comprehension below should be faster
        for rno in range(len(test_df['id'])):
            test_df.loc[(test_df['id'] == test_df['id'][rno]) & (preds_flat[rno] == 1), 'prediction'] = 'biased'
            test_df.loc[(test_df['id'] == test_df['id'][rno]) & (preds_flat[rno] == 0), 'prediction'] = 'unbiased'
            # sentence-level bipol for biased samples only - iterate through the dict of axes and calculate
            if preds_flat[rno] == 1:
                #print(" Prediction is 1 - BIASED", rno)
                for a_k, a_v in axes.items():
                    list_shade = []                                                 # initialize
                    for b in range(len(a_v)):                                       # Go to next dict in list
                        inshade_cnt = 0
                        for k in a_v[b].keys():
                            a_v[b][k] = test_df['comment_text'][rno].count(k)       # In future make the column name dynamic
                            axes2[a_k][b][k] = axes2[a_k][b][k] + a_v[b][k]         # sum all sensitive tokens for explainability
                            #print(a_v[b][k])
                            inshade_cnt += a_v[b][k]
                        #print(inshade_cnt)
                        list_shade.append(inshade_cnt)
                    #print('Shades ', list_shade)
                    bip_divis = sum(list_shade)
                    if len(list_shade) < 1:
                        bip_sent1 = 0
                    else:
                        bip_sent1 = max(list_shade)
                        list_shade.remove(bip_sent1)
                    if len(list_shade) < 1:                                         # check empty cases
                        bip_sent2 = 0
                    else:
                        bip_sent2 = max(list_shade)                                 # get max after the 1st max
                    if bip_divis > 0:                                           # To avoid divide by zero error
                        bip_sent = abs(bip_sent1 - bip_sent2) / bip_divis
                    bipol_sent += bip_sent                                       # sum over axes
                
                #print('Total over axis ', bipol_sent, len(axes))
                bipol_sent = bipol_sent/len(axes)                                    # average over axes per sentence
                #print('After division - Total over axis ', bipol_sent)
                bipol_sent_agg += bipol_sent                                        # sum over multiple sentences
                #print('Total over sentences ', bipol_sent_agg)
                bipol_sent = 0                                                      # re-initialize for next sentence
        
        test_df_ = test_df[test_df['prediction'] == 'biased']                       # Find out the total of biased samples
        print(test_df_.head())
        print('biased total no: ', len(test_df_))
        if len(test_df_) > 0:
            bipol_sent_agg = bipol_sent_agg/len(test_df_)                           # Overall sentence-level bipol
            print('After division - Total over sentences ', bipol_sent_agg)
        else:
            bipol_sent_agg = 0                                                      # return zero for <= 0 (hopefully shouldn't be)

        if 'label' in test_df2.columns:                                             # In future, make this column dynamic
            test_df2['label'] = test_df2.label.replace(label_dict)                  # replace labels with their nos for F1 scores
            tvals = test_df2['label'].values.tolist()

    eval_time_elapsed = time.time() - eval_start_time

    ### Check if gold labels are provided
    if len(tvals) > 0:
        f1, f1_w, f1_m = f1_score_func(tvals, preds_flat)
        tn, fp, fn, tp = confusion_matrix_func(tvals, preds_flat)
        print('All four - tn, fp, fn, tp: ', tn, fp, fn, tp)
        # Let the metric check for FP to remove them for datasets that have labels.
        bipol_cop = (tp + fp)/(tn + fp + fn + tp)
    else:
        print('Biased no, Total no ', len(test_df_), len(test_df))
        bipol_cop = len(test_df_) / len(test_df)                                   # alternative calculation when there's no ground truth labels
    bipol = bipol_cop * bipol_sent_agg

    # ### explain the score
    print("Some explanation: It does not say if the bias is towards or against a group:\n")
    message = ''
    if bipol == 0:
        message = f'The data seems to have little or no bias, though this may be due to the quality of lexicon used. The following axes have the following counts, in addition to {args.model_name} classification:' 
        print(message)
        print(f'{axes2}')
    elif bipol > 0 and bipol < 0.2:
        message = f'The data seems to have mild bias because the following axes have the following counts, in addition to {args.model_name} classification:'
        print(message)
        print(f'{axes2}')
    elif bipol >= 0.2 and bipol < 0.4:
        message = f'The data seems to have moderate bias because the following axes have the following counts, in addition to {args.model_name} classification:'
        print(message)
        print(f'{axes2}')
    elif bipol >= 0.4 and bipol < 0.6:
        message = f'The data seems to have broad bias because the following axes have the following counts, in addition to {args.model_name} classification:'
        print(message)
        print(f'{axes2}')
    elif bipol >= 0.6 and bipol < 0.8:
        message = f'The data seems to have substantial bias because the following axes have the following counts, in addition to {args.model_name} classification:'
        print(message)
        print(f'{axes2}')
    elif bipol >= 0.8:
        message = f'The data seems to have extreme bias because the following axes have the following counts, in addition to {args.model_name} classification:'
        print(message)
        print(f'{axes2}')
    
    pred_ts = 'ijcnn_' + args.data_folder + args.modeldir.split('/')[-2]                              # for filename to save
    print(' ')
    print(f'Evaluation time: {eval_time_elapsed}')
    if len(tvals) > 0:
        print(f'F1: {f1}, weighted F1: {f1_w}, macro F1: {f1_m}, bipol corpus-level: {bipol_cop}, bipol sentence-level: {bipol_sent_agg}, bipol: {bipol}')
        with open(pred_ts+ '_results.txt', "a+") as f:
            s = f.write(f'Evaluation time: {eval_time_elapsed}, F1: {f1}, weighted F1: {f1_w}, macro F1: {f1_m}, All - tn, fp, fn, tp: {tn, fp, fn, tp}, bipol corpus-level: {bipol_cop}, bipol sentence-level: {bipol_sent_agg}, bipol: {bipol}')
            s = f.write(f'\n\n{message}')
            s = f.write(f'\n\n{axes2}')
    else:
        print(f'bipol corpus-level: {bipol_cop}, bipol sentence-level: {bipol_sent_agg}, bipol: {bipol}')
        with open(pred_ts+ '_results.txt', "a+") as f:
            s = f.write(f'Evaluation time: {eval_time_elapsed}, bipol corpus-level: {bipol_cop}, bipol sentence-level: {bipol_sent_agg}, bipol: {bipol}')
            s = f.write(f'\n\n{message}')
            s = f.write(f'\n\n{axes2}')
    test_df.to_csv(pred_ts  + '_output.csv', index=False)
