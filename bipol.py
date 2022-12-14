import logging
from statistics import mean

import pandas as pd
from sklearn.metrics import accuracy_score

import wandb
from simpletransformers.classification import ClassificationArgs, ClassificationModel
#from utils import load_rte_data_file

sweep_config = {
    "name": "vanilla-sweep-batch-16",
    "method": "bayes",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "num_train_epochs": {"min": 1, "max": 2},
        "learning_rate": {'max': 0.1, 'min': 0.0001}, #{"min": 0.0, "max": 4e-4},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 6,},
}

sweep_id = wandb.sweep(sweep_config, project="test - RoBERTa")


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def load_rte_data_file(filepath):
    df = pd.read_json(filepath, lines=True)
    df = df.rename(columns={"premise": "text_a", "hypothesis": "text_b", "label": "labels"})
    df = df[["text_a", "text_b", "labels"]]
    return df


# # #### Binary classification
# prefix = 'data/bi_yelp_review/'

# binary_train_df = pd.read_csv(prefix + 'train.csv', header=None)
# #print(binary_train_df.head())

# binary_eval_df = pd.read_csv(prefix + 'test.csv', header=None)
# #print(binary_eval_df.head())

# binary_train_df[0] = (binary_train_df[0] == 2).astype(int)      # make 1 = 0 & 2 = 1
# binary_eval_df[0] = (binary_eval_df[0] == 2).astype(int)

# binary_train_df = pd.DataFrame({
#     'prefix': ["binary classification" for i in range(len(binary_train_df))],
#     'input_text': binary_train_df[1].str.replace('\n', ' '),
#     'target_text': binary_train_df[0].astype(str),
# })

# #print(binary_train_df.head())

# binary_eval_df = pd.DataFrame({
#     'prefix': ["binary classification" for i in range(len(binary_eval_df))],
#     'input_text': binary_eval_df[1].str.replace('\n', ' '),
#     'target_text': binary_eval_df[0].astype(str),
# })

# train_df = binary_train_df
# eval_df = binary_eval_df
# test_df = eval_df


# Preparing train data
train_df = load_rte_data_file("data/rte/train.jsonl")
eval_df = pd.read_json("data/rte/val.jsonl", lines=True, orient="records")
test_df = pd.read_json("data/rte/test.jsonl", lines=True, orient="records")

train_df = train_df[:20]
eval_df = train_df #eval_df[:10]
test_df = train_df #test_df[:10]


# train_df = train_df[train_df["text_a"].str.contains("Christopher Reeve had an") == False]
# train_df = train_df[train_df["text_b"].str.contains("Christopher Reeve had an") == False]
# eval_df = eval_df[eval_df["premise"].str.contains("Christopher Reeve had an") == False]
# eval_df = eval_df[eval_df["hypothesis"].str.contains("Christopher Reeve had an") == False]


model_args = ClassificationArgs()
model_args.eval_batch_size = 32
model_args.evaluate_during_training = True
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = 1000
model_args.evaluate_during_training_steps = -1
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.learning_rate = 1e-5
model_args.manual_seed = 4
model_args.max_seq_length = 256
model_args.multiprocessing_chunksize = 5000
model_args.no_cache = True
model_args.no_save = True
model_args.num_train_epochs = 1
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 2
model_args.labels_list = ["not_entailment", "entailment"]
model_args.output_dir = "outputs"
model_args.best_model_dir = "outputs/best_model"
model_args.wandb_project = "test - RoBERTa"
#model_args.wandb_kwargs = sweep_config #{"name": "default"}


def train():
    wandb.init()
    # Create a TransformerModel
    model = ClassificationModel("roberta", "roberta-base", use_cuda=True, args=model_args)

    # Train the model
    model.train_model(
        train_df,
        eval_df=eval_df,
        accuracy=lambda truth, predictions: accuracy_score(
            truth, [round(p) for p in predictions]
        ),
    )

    wandb.join()

    #model.eval_model(test_df, verbose=True)
wandb.agent(sweep_id, train)
