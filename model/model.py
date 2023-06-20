from setfit import SetFitModel, SetFitTrainer, sample_dataset
from datasets import load_dataset, Dataset
from sentence_transformers.losses import CosineSimilarityLoss

import torch


import pandas as pd
import seaborn as sns

# Read Data -> Dataset used here is the twitter extracted sentiment dataset from Kaggle
trainDS = pd.read_csv('train.csv')
trainDS.sample(10)

# replacing target column from text to numerical values.
replacement_dict = {'positive': 1, 'negative': 2, 'neutral': 0}
trainDS['sentiment'] = trainDS['sentiment'].replace(replacement_dict)

# plotting target column and classes
# sns.countplot(trainDS,x='sentiment')
# print(trainDS.shape)

sampleDS = trainDS[:100]   # 100 sample taken to train
sampleDS.to_csv('sampleTrainDS.csv')

valDS = trainDS[27460:]     # last 20 sample taken for validation
valDS.to_csv('sampleValDS.csv')

# Load Dataset
dataset = load_dataset('csv', data_files={
    'train':['sampleTrainDS.csv'],
    'eval':['sampleValDS.csv']},
)

# Model from Huggingface
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Model Trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['eval'],
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=16,
    num_iterations=20,  # The number of text pairs to generate for contrastive learning
    num_epochs=1,  # The number of epochs to use for contrastive learning
    column_mapping={"text": "text", "sentiment": "label"}  # Map dataset columns to text/label expected by trainer
)

# Training the model
trainer.train()

# evaluating model on our dataset
metrics = trainer.evaluate()
print(metrics)


# save trained model
trainer.model._save_pretrained(save_directory="testing")