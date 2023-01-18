import pandas as pd
import re
from transformers import BertTokenizer, BertForMaskedLM, pipeline
from nltk.stem import PorterStemmer
ps = PorterStemmer()


print("Downloading BERT model...")
#Bert config
BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
BERT_base = BertForMaskedLM.from_pretrained('bert-base-cased', return_dict=True)
print("Done!")

#pipeline object for fill-mask task
unmasker = pipeline('fill-mask', model= BERT_base , tokenizer= BERT_tokenizer)


#create df for the ranked predictions + incorrect predictions
rank_df =pd.DataFrame(index=['Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5', 'Incorrect', 'Tot.'])


print("Uploading the prompt dataframe...")
prompt_df = pd.read_csv('/Users/caput/ELLie-ellipsis_and_thematic_fit_with_TLMs/Task3_Verb_and_dObj_retrieval/ELLie_Prompts.csv', sep =",", index_col = 0)
prompt_df['Prompt_BERT'] = prompt_df['Prompt_BERT'].astype('str')
print("Done!")


def ranked_predictions(df):
    Rank1 = 0
    Rank2 = 0
    Rank3 = 0
    Rank4 = 0
    Rank5 = 0
    wrong_prediction = 0
    Tot = 0

    for index, row in df.iterrows():
        # take the prompt for that sentence
        prompt_BERT = row['Prompt_BERT']

        # take the target Verb to retrieve (base form of the verb)
        targetW = row['Verb_to_retrieve']

        # retrieve the verb stem
        stemmed_word = ps.stem(targetW)

        # use this regex to retrieve every form of the verb, basing on its stem
        # (e.g. 'complete' - stem:complet - 'completing, completed, completes)
        regex = rf"\b{stemmed_word}[a-z]*\b"

        #list of the 5 most probable filler found by BERT to fill the mask token
        fillers = unmasker(prompt_BERT)

        #The filler list will contain the five most probable fillers identified by BERT
        # in order of probability (from most likely to least likely)
        filler_list = []
        for fill in fillers:
            filler_list.append(fill['token_str'])


        #if the correct token is the 1st most probable BERT-filler
        if re.findall(regex, filler_list[0], re.I):
            Rank1 += 1
        #if the correct token is the 2nd most probable BERT-filler
        elif re.findall(regex, filler_list[1], re.I):
            Rank2 += 1
        #if the correct token is the 3rd most probable BERT-filler
        elif re.findall(regex, filler_list[2], re.I):
            Rank3 += 1
        #if the correct token is the 4th most probable BERT-filler
        elif re.findall(regex, filler_list[3], re.I):
            Rank4 += 1
        #if the correct token is the 5th most probable BERT-filler
        elif re.findall(regex, filler_list[4], re.I):
            Rank5 += 1
        #if the correct token is not in the 5 most probable BERT-fillers:
        # BERT is not able to retrieve the correct elided verb
        else:
            wrong_prediction += 1
        #Tot prediction
        Tot += 1


    ranking_score = [Rank1 , Rank2, Rank3, Rank4 , Rank5, wrong_prediction, Tot]
    rank_df['Predictions'] = ranking_score
    return rank_df



rank_df = ranked_predictions(prompt_df)

rank_df.to_csv(
        '/Users/caput/ELLie-ellipsis_and_thematic_fit_with_TLMs/outputs_csv/Ranked_predictions_verb_retrieval_BERT.csv')
print("Dataset with ranked predictions correctly saved!")
