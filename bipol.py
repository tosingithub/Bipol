import logging
from statistics import mean
import argparse
import pandas as pd
from sklearn.metrics import f1_score
import wandb
from simpletransformers.classification import ClassificationArgs, ClassificationModel
import re
import utility


#from utils import load_rte_data_file

jig_folder = '/home/shared_data/bipol/Jigsaw_kaggle/'
sbic_folder = '/home/shared_data/bipol/sbicv2/'
new_folder = '/home/shared_data/bipol/new/'
mab_swedish = '/home/shared_data/bipol/mab_swedish/'

### If run from CLI, you may change the 2 default arguments below.
parser = argparse.ArgumentParser(description='Bias Detection')
parser.add_argument('--data_folder', type=str, default=mab_swedish, help='location of the data')     # of sbic_folder
parser.add_argument('--model_name', type=str, default='sv_bert', help='name of the deep model')     # or deberta
args = parser.parse_args()

sweep_config = {
    "name": "bias-sweep-batch-16",
    "method": "bayes",
    "metric": {"name": "f1", "goal": "maximize"},
    "parameters": {
        "num_train_epochs": {"min": 6, "max": 10},
        "learning_rate": {'max': 0.001, 'min': 0.00002}, #{"min": 0.0, "max": 4e-4},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 6,},
}

sweep_id = wandb.sweep(sweep_config, project="Bias Prediction")


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
elif args.data_folder == mab_swedish:
    train_df = pd.read_csv(mab_swedish + 'swedish_mab_train.csv', header=0)
    train_df = utility.preprocess_pandas(train_df, list(train_df.columns))
    eval_df = pd.read_csv(mab_swedish + 'swedish_mab_val.csv', header=0)
    eval_df = utility.preprocess_pandas(eval_df, list(eval_df.columns))
    test_df = pd.read_csv(mab_swedish + 'swedish_mab_test.csv', header=0)
    test_df = utility.preprocess_pandas(test_df, list(test_df.columns))


model_args = ClassificationArgs()
model_args.eval_batch_size = 32
model_args.evaluate_during_training = True
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = 1000
model_args.evaluate_during_training_steps = -1
model_args.save_eval_checkpoints = True
model_args.save_model_every_epoch = True
#model_args.learning_rate = 1e-5
# model_args.manual_seed = 4
model_args.max_seq_length = 256
model_args.multiprocessing_chunksize = 5000
model_args.no_cache = True
model_args.no_save = False
#model_args.num_train_epochs = 1
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.train_batch_size = 32 # 16
model_args.gradient_accumulation_steps = 2
model_args.labels_list = ["biased", "unbiased"]
model_args.output_dir = "outputs"
model_args.best_model_dir = "outputs/best_model"
model_args.wandb_project = "Bias Prediction"
#model_args.wandb_kwargs = sweep_config #{"name": "default"}


def train():
    wandb.init()

    if args.model_name == 'roberta':
        model = ClassificationModel("roberta", "roberta-base", use_cuda=True, args=model_args)
    elif args.model_name == 'deberta':
        model = ClassificationModel("deberta", "microsoft/deberta-base", use_cuda=True, args=model_args)
    elif args.model_name == 'electra':
        model = ClassificationModel("electra", "google/electra-base-generator", use_cuda=True, args=model_args)
    elif args.model_name == 'sv_bert':
        model = ClassificationModel('bert', 'KB/bert-base-swedish-cased', use_cuda=True, args=model_args)


    # Train the model
    model.train_model(
        train_df,
        eval_df=eval_df,
        f1=lambda truth, predictions: f1_score(
            truth, [round(p) for p in predictions]
        ),
    )

    wandb.join()

    #model.eval_model(test_df, verbose=True)
wandb.agent(sweep_id, train, count=5)
