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
import numpy as np
import copy
import os
import time
from datasets import load_dataset


#from utils import load_rte_data_file

### --data_folder
jig_folder = '/home/shared_data/bipol/Jigsaw_kaggle/'
sbic_folder = '/home/shared_data/bipol/sbicv2/'
new_folder = '/home/shared_data/bipol/new/'

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



### If run from CLI, you may change the 2 default arguments below.
parser = argparse.ArgumentParser(description='Bias Detection')
parser.add_argument('--data_folder', type=str, default=sbic_folder, help='location of the data')     # of sbic_folder
parser.add_argument('--model_name', type=str, default='roberta', help='name of the deep model')     # or deberta
parser.add_argument('--axes_folder', type=str, default='axes/', help='name of the folder containing the bias axes files')     # or deberta
parser.add_argument('--modeldir', type=str, default=rob_jig, help='directory of the model checkpoint')

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing the data
if args.data_folder == jig_folder:
    train_df = pd.read_csv(jig_folder + 'jig_train.csv', header=0)
    train_df = utility.preprocess_pandas(train_df, list(train_df.columns))
    eval_df = pd.read_csv(jig_folder + 'jig_val.csv', header=0)
    eval_df = utility.preprocess_pandas(eval_df, list(eval_df.columns))
    test_df = pd.read_csv(jig_folder + 'jig_test.csv', header=0)
    test_df = utility.preprocess_pandas(test_df, list(test_df.columns))
elif args.data_folder == sbic_folder:
    train_df = pd.read_csv(sbic_folder + 'sbic_train.csv', header=0)
    train_df = utility.preprocess_pandas(train_df, list(train_df.columns))
    eval_df = pd.read_csv(sbic_folder + 'sbic_val.csv', header=0)
    eval_df = utility.preprocess_pandas(eval_df, list(eval_df.columns))
    test_df = pd.read_csv(sbic_folder + 'sbic_test.csv', header=0)
    test_df = utility.preprocess_pandas(test_df, list(test_df.columns))
elif args.data_folder == new_folder:
    train_df = pd.read_csv(new_folder + 'new_train.csv', header=0)
    train_df = utility.preprocess_pandas(train_df, list(train_df.columns))
    eval_df = pd.read_csv(new_folder + 'new_val.csv', header=0)
    eval_df = utility.preprocess_pandas(eval_df, list(eval_df.columns))
    test_df = pd.read_csv(new_folder + 'new_test.csv', header=0)
    test_df = utility.preprocess_pandas(test_df, list(test_df.columns))
elif args.data_folder == 'squad':
    train_df = load_dataset('squad_v2', split='validation')
    print(train_df['context'])                                           # returns a list
elif args.data_folder == 'imdb':
    train_df = load_dataset('imdb', split='validation')

# model_args = ClassificationArgs()
# model_args.eval_batch_size = 32
# model_args.evaluate_during_training = True
# model_args.evaluate_during_training_silent = False
# model_args.evaluate_during_training_steps = 1000
# model_args.evaluate_during_training_steps = -1
# model_args.save_eval_checkpoints = True
# model_args.save_model_every_epoch = True
# ###change the next two below for test set
# model_args.learning_rate = 0.000305675810646798
# model_args.num_train_epochs = 8

# model_args.manual_seed = 4
# model_args.max_seq_length = 256
# model_args.multiprocessing_chunksize = 5000
# model_args.no_cache = True
# model_args.no_save = False
# model_args.overwrite_output_dir = True
# model_args.reprocess_input_data = True
# model_args.train_batch_size = 16
# model_args.gradient_accumulation_steps = 2
# model_args.labels_list = ["biased", "unbiased"]
# model_args.output_dir = "test_outputs"
# model_args.best_model_dir = "test_outputs/best_model"
# model_args.wandb_project = "Testset- Bias Prediction"
# #model_args.wandb_kwargs = sweep_config #{"name": "default"}

##### Usual Prediction Code

def f1_score_func(labels, preds_flat):
    return f1_score(labels, preds_flat, average=None), f1_score(labels, preds_flat, average="weighted"), f1_score(labels, preds_flat, average="macro")

def confusion_matrix_func(labels, preds_flat):
    print(confusion_matrix(labels, preds_flat))  # cell(r,c) 0,0- TN, 1,0- FN, 1,1- TP, 0,1- FP
    tn, fp, fn, tp = confusion_matrix(labels, preds_flat).ravel()
    return tn, fp, fn, tp

def prec_rec(labels):
    precision_recall_curve()

# def eval_test():
#     #wandb.init()

#     if args.model_name == 'roberta':
#         model = ClassificationModel("roberta", "roberta-base", use_cuda=True, args=model_args)
#     elif args.model_name == 'deberta':
#         model = ClassificationModel("deberta", "microsoft/deberta-base", use_cuda=True, args=model_args)
#     elif args.model_name == 'electra':
#         model = ClassificationModel("electra", "google/electra-base-generator", use_cuda=True, args=model_args)


#     # Train the model
#     model.train_model(
#         train_df,
#         eval_df=eval_df,
#         f1=lambda truth, predictions: f1_score(
#             truth, [round(p) for p in predictions]
#         ),
#     )

#     model.eval_model(test_df, verbose=True)

if __name__=="__main__":
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
        #test_df = test_df[:8]          # small for trials
        print('Total no: ', len(test_df))

        test_df2 = test_df.copy()                                           # test_df2 for label replacement for roberta below
        print(test_df.head())
        if not 'prediction' in test_df.columns:
            test_df.insert(len(test_df.columns), 'prediction', '')
        
        label_dict = {}                                                     # For associating raw labels with indices/nos
        #pred_ts = args.model_name.split('/')[-1] + '_output.csv'            # filename of output file
        possible_labels = train_df.label.unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)                                                   # {'unbiased': 0, 'biased': 1}
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
        #pred_ts = args.model_name.split('/')[-1] + '_output.csv'            # filename of output file
        possible_labels = train_df.label.unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)                                                   # {'unbiased': 0, 'biased': 1}
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
        #pred_ts = args.model_name.split('/')[-1] + '_output.csv'            # filename of output file
        possible_labels = train_df.label.unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)                                                   # {'unbiased': 0, 'biased': 1}
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
    
    pred_ts = args.modeldir.split('/')[-2]                              # for filename to save
    print(' ')
    print(f'Evaluation time: {eval_time_elapsed}')
    if len(tvals) > 0:
        print(f'F1: {f1}, weighted F1: {f1_w}, macro F1: {f1_m}, All - tn, fp, fn, tp: {tn, fp, fn, tp}, bipol corpus-level: {bipol_cop}, bipol sentence-level: {bipol_sent_agg}, bipol: {bipol}')
        with open(pred_ts+ '_results.txt', "a+") as f:
            s = f.write(f'Evaluation time: {eval_time_elapsed}, F1: {f1}, weighted F1: {f1_w}, macro F1: {f1_m}, bipol corpus-level: {bipol_cop}, bipol sentence-level: {bipol_sent_agg}, bipol: {bipol}')
            s = f.write(f'\n\n{message}')
            s = f.write(f'\n\n{axes2}')
    else:
        print(f'bipol corpus-level: {bipol_cop}, bipol sentence-level: {bipol_sent_agg}, bipol: {bipol}')
        with open(pred_ts+ '_results.txt', "a+") as f:
            s = f.write(f'Evaluation time: {eval_time_elapsed}, bipol corpus-level: {bipol_cop}, bipol sentence-level: {bipol_sent_agg}, bipol: {bipol}')
            s = f.write(f'\n\n{message}')
            s = f.write(f'\n\n{axes2}')
    test_df.to_csv(pred_ts  + '_output.csv', index=False)
