import statistics
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import minicons
from minicons import scorer

print("Downloading models...")
#download the TLanguage_Models using the scorer package from minicons_lib
gpt_model = scorer.IncrementalLMScorer('gpt2-large', 'cpu')
bert_model = scorer.MaskedLMScorer('bert-base-cased', 'cpu')
print("Done!")

#use minicons_lib to compute sentence log_probability ( model.sequence_score() ) with different TLMs.
#Since the dataset is composed of sentences of different lengths, we normalize the log_prob by number of tokens through the reduction parameter
def sent_logProb(sent, TLM):
    if TLM == "GPT":
        #the score is the ln of sentence probability
        sentProb = gpt_model.sequence_score(sent, reduction = lambda x: x.mean(0).item())[0]
        return sentProb
    elif TLM == "BERT":
        sentProb = bert_model.sequence_score(sent,reduction = lambda x: x.mean(0).item())[0]
        return sentProb

#compute sentence log_prob for each sentence of the dataset
#create a new column in the dataset.
#Add probability score to that column for each row
def create_sentProb_col(df, TLM):
    if TLM == "GPT":
        for index, row in df.iterrows():
            sent = row['Sentence']
            sentence_prob = sent_logProb(sent, "GPT")
            df.loc[index, 'Sentence_score_(GPT)'] = sentence_prob
        # convert value of the column to float values (otherwise it is always 0.0)
        #df = df.astype({'Sentence_score_(GPT)': 'float'})
    elif TLM == "BERT":
        for index, row in df.iterrows():
            sent = row['Sentence']
            sentence_prob = sent_logProb(sent, "BERT")
            df.loc[index, 'Sentence_score_(BERT)'] = sentence_prob
        # convert value of the column to float values (otherwise it is always 0.0)
        #df = df.astype({'Sentence_score_(BERT)': 'float'})





#load the ellipsis dataset
ellipsis_df = pd.read_csv('/Users/caput/ELLie-ellipsis_and_thematic_fit_with_TLMs/ELLie.csv', sep = ",")

print("Dataset uploaded!")

#convert 'Sentence' columns to string type
ellipsis_df= ellipsis_df.astype({'Sentence': 'string'})

print("Computing sentence probabilities with GPT-2...")
#compute sentence prob with GPT-model and add the score to a new column
create_sentProb_col(ellipsis_df, "GPT")
print("Done!")

print("Computing sentence probabilities with BERT...")
#compute sentence prob with BERT-model and add the score to a new column
create_sentProb_col(ellipsis_df, "BERT")
print("Done!")


ellipsis_df.to_csv('/Users/caput/ELLie-ellipsis_and_thematic_fit_with_TLMs/outputs_csv/ELLie_with_Sentence_Prob.csv')
print("Dataset with sentence probabilities saved!")