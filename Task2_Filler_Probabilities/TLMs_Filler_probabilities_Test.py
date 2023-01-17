
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import minicons
from minicons import scorer


#retrieve filler probabilities by token-position
def filler_prob(df):
    for index, row in df.iterrows():

        #all semantic roles but not Agent ones
        if row['Semantic_Role'] != 'Agent':
            sent = row['Sentence']
            standard_pos_gpt = row['Standard_Filler_position_GPT']
            standard_pos_bert= row['Standard_Filler_position_BERT']
            elliptical_pos_gpt = row['Elliptical_Filler_position_GPT']
            elliptical_pos_bert = row['Elliptical_Filler_position_BERT']

            word_probs_gpt = gpt_model.token_score(sent)
            word_probs_bert = bert_model.token_score(sent)

            #retrieve the string of the filler in the standard part of the sentence
            df.loc[index, 'Standard_filler'] = word_probs_bert[0][standard_pos_bert][0]
            # retrieve the string of the filler in the elliptical part of the sentence
            df.loc[index, 'Elliptical_filler'] = word_probs_bert[0][elliptical_pos_bert][0]



            # compute prob of the filler with BERT
            df.loc[index, 'Score1_BERT'] = word_probs_bert[0][standard_pos_bert][1]
            # compute prob of the filler with BERT
            df.loc[index, 'Score2_BERT'] = word_probs_bert[0][elliptical_pos_bert][1]

            # compute prob of the filler with GPT
            df.loc[index, 'Score1_GPT'] = word_probs_gpt[0][standard_pos_gpt][1]
            # compute prob of the filler with GPT
            df.loc[index, 'Score2_GPT'] = word_probs_gpt[0][elliptical_pos_gpt][1]


        # Agent fillers are computed only with BERT since GPT is a causal LM and it cannot see what is on the right context
        else:
            sent = row['Sentence']
            standard_pos_bert = row['Standard_Filler_position_BERT']
            elliptical_pos_bert = row['Elliptical_Filler_position_BERT']

            word_probs_bert = bert_model.token_score(sent)

            df.loc[index, 'Standard_filler'] = word_probs_bert[0][standard_pos_bert][0]
            df.loc[index, 'Elliptical_filler'] = word_probs_bert[0][elliptical_pos_bert][0]

            df.loc[index, 'Score1_BERT'] = word_probs_bert[0][standard_pos_bert][1]
            df.loc[index, 'Score2_BERT'] = word_probs_bert[0][elliptical_pos_bert][1]






#load the ellipsis dataset with semantic roles (filler) token-position (both for GPT tokenizer and BERT tokenizer)
ellipsis_df = pd.read_csv('/Users/caput/ELLie-ellipsis_and_thematic_fit_with_TLMs/Task2_Filler_Probabilities/ELLie_with_filler_positions.csv', sep = ",")

print("Dataset uploaded!")

#convert 'Sentence' columns to string type
ellipsis_df= ellipsis_df.astype({'Semantic_Role': 'string'})

print("Downloading models...")
gpt_model = scorer.IncrementalLMScorer('gpt2-large', 'cpu')
bert_model = scorer.MaskedLMScorer('bert-base-cased', 'cpu')

print("Done!")


filler_prob(ellipsis_df)

print("Probabilities retrieved!")


ellipsis_df.to_csv('/Users/caput/ELLie-ellipsis_and_thematic_fit_with_TLMs/outputs_csv/ELLie_with_Filler_Prob.csv')
print("Dataset with filler probabilities correctly saved!")

