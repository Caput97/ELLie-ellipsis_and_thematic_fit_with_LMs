import pandas as pd
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed
from transformers import BertTokenizer, BertForMaskedLM
from nltk.stem import PorterStemmer
ps = PorterStemmer()

#GPT conf

GPT2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
GPT2 = GPT2LMHeadModel.from_pretrained('gpt2-large', return_dict=True)
text_generator = pipeline('text-generation', model = GPT2, tokenizer = GPT2_tokenizer)


#BERT conf

BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
BERT_base = BertForMaskedLM.from_pretrained('bert-base-cased', return_dict=True)
unmasker = pipeline('fill-mask', model= BERT_base , tokenizer= BERT_tokenizer)

print("Models downloaded!")
print()



prompt_df = pd.read_csv('/Users/caput/ELLie-ellipsis_and_thematic_fit_with_TLMs/Task3_Verb_and_dObj_retrieval/ELLie_Prompts.csv', sep =",", index_col = 0)

prompt_df['Sentence'] = prompt_df['Sentence'].astype('str')
prompt_df['Prompt_GPT'] = prompt_df['Prompt_GPT'].astype('str')
prompt_df['Verb_to_retrieve'] = prompt_df['Verb_to_retrieve'].astype('str')
prompt_df['DObj_to_retrieve'] = prompt_df['DObj_to_retrieve'].astype('str')

print("Dataset correctly uploaded!")
print()

#create df for the accuracy scores
accuracy_df =pd.DataFrame(index=[ 'T-T', 'T-AT', 'AT-T', 'AT-AT', 'T-SP viol', 'Tot.'])



#function to create 3 new sentences based on prompt input. (Text-generation task with sampling technique)
def text_generation_GPTsampling(df, prompt):
    set_seed(50)
    generated_text1 = text_generator(prompt, max_new_tokens=4, temperature=0.7, top_k=3, top_p=0.92, num_return_sequences=3)[0][
        'generated_text']
    generated_text2 = text_generator(prompt, max_new_tokens=4, temperature=0.7, top_k=3, top_p=0.92, num_return_sequences=3)[1][
        'generated_text']
    generated_text3 = text_generator(prompt, max_new_tokens=4, temperature=0.7, top_k=3, top_p=0.92, num_return_sequences=3)[2][
        'generated_text']

    return generated_text1, generated_text2, generated_text3



#function to retrieve the elided verb of the sentence.
# Given the elliptical sentence and prompt, the model has to retrieve the mask filler.
def fill_mask_BERT(df, prompt):

    filler = unmasker(prompt)[0]['token_str']

    return filler


#match condition to retrieve the verb with regex in generated text (with sampling technique)
def Verb_match_condition_sampling(regex, text1, text2, text3, sentence):

    match_1 = False
    match_2 = False
    acceptable_match1 = False
    acceptable_match2 = False

    #if in the generated part of the prompt there are two matches for that regex
    if (len(re.findall(regex, text1, re.I)) == 2) \
            or (len(re.findall(regex, text2, re.I)) == 2) \
            or (len(re.findall(regex, text3, re.I)) == 2):
        match_2 = True

    #if in the generated part of the prompt there is only one match for that regex
    if (len(re.findall(regex, text1, re.I)) == 1) \
            or (len(re.findall(regex, text2, re.I)) == 1) \
            or (len(re.findall(regex, text3, re.I)) == 1):
        match_1 = True

    # if the matched verb is present in the same verbal form both in the original elliptical sentence
    # and in the generated part of the prompt
    if re.findall(regex, sentence, re.I) and match_2: acceptable_match2 = True

    # if the matched verb is present only in the generated part of the prompt but not in the original elliptical sentence
    # it is the case of two verbal form of the same irregular verb.
    #e.g. elliptical sentence: 'drank' - generated sentence 'drink'
    if not re.findall(regex, sentence, re.I) and match_1: acceptable_match1 = True

    return acceptable_match1, acceptable_match2


#match condition to retrieve the direct object with regex in generated text (with sampling technique)
def dObj_match_condition_sampling(regex,text1, text2, text3, sentence ):

    match_2 = False
    acceptable_match = False

    if (len(re.findall(regex, text1, re.I)) == 2) \
            or (len(re.findall(regex, text2, re.I)) == 2) \
            or (len(re.findall(regex, text3, re.I)) == 2):
        match_2 = True

    if re.findall(regex, sentence, re.I) and match_2: acceptable_match = True

    return acceptable_match


#match condition to retrieve the verb with regex in generated text (without sampling technique)
def Verb_match_condition(regex, text, sentence):
    match_1 = False
    match_2 = False
    acceptable_match1 = False
    acceptable_match2 = False

    if (len(re.findall(regex, text, re.I)) == 2) :
        match_2 = True

    if (len(re.findall(regex, text, re.I)) == 1) :
        match_1 = True

    if re.findall(regex, sentence, re.I) and match_2: acceptable_match2 = True
    if not re.findall(regex, sentence, re.I) and match_1: acceptable_match1 = True

    return acceptable_match1, acceptable_match2


#match condition to retrieve the verb with regex in generated text (without sampling technique)
def dObj_match_condition(regex, text, sentence):
    match_2 = False
    acceptable_match = False

    if (len(re.findall(regex, text, re.I)) == 2):
        match_2 = True

    if re.findall(regex, sentence, re.I) and match_2: acceptable_match = True

    return acceptable_match



#function to compute verb and dObj retrieval score given the prompt
#GPT-2 is used with a text-generation task
#BERT is used with a fill-mask task (only for the verb)
def score(df, model, sampling):
    accuracy = 0
    dObj_accuracy = 0
    accuracy_TT = 0
    dObj_accuracy_TT = 0
    accuracy_TAT = 0
    dObj_accuracy_TAT = 0
    accuracy_ATT = 0
    dObj_accuracy_ATT = 0
    accuracy_ATAT = 0
    dObj_accuracy_ATAT = 0
    accuracy_Tviol = 0
    dObj_accuracy_Tviol = 0

    if model == 'GPT' and sampling == True:

        for index, row in df.iterrows():
            # take the prompt for that sentence
            prompt_GPT2 = row['Prompt_GPT']

            generated_text1, generated_text2, generated_text3 = text_generation_GPTsampling(df, prompt_GPT2)

            df.loc[index, 'Generated_text_GPTsampling_1'] = generated_text1
            df.loc[index,'Generated_text_GPTsampling_2'] = generated_text2
            df.loc[index,'Generated_text_GPTsampling_3'] = generated_text3



            # take the target Verb to retrieve (base form of the verb)
            targetV = row['Verb_to_retrieve']

            # retrieve the verb stem
            stemmed_V = ps.stem(targetV)

            # use this regex to retrieve every form of the verb, basing on its stem
            # (e.g. 'complete' - stem:complet - 'completing, completed, completes)
            regexV = rf"\b{stemmed_V}[a-z]*\b"

            acceptability1, acceptability2 = Verb_match_condition_sampling(regexV, generated_text1, generated_text2,generated_text3, row['Sentence'])

            if acceptability1 or acceptability2:
                accuracy += 1
                if row['Condition'] == 'T - T':
                    accuracy_TT += 1
                elif row['Condition'] == 'T - AT':
                    accuracy_TAT += 1
                elif row['Condition'] == 'AT - T':
                    accuracy_ATT += 1
                elif row['Condition'] == 'AT - AT':
                    accuracy_ATAT += 1
                elif row['Condition'] == 'T - SP violation':
                    accuracy_Tviol += 1

                df.loc[index,'Match_verb_GPTsampling'] = "Yes"
            else:
                df.loc[index,'Match_verb_GPTsampling'] = "No"


            score_values = [accuracy_TT, accuracy_TAT, accuracy_ATT, accuracy_ATAT, accuracy_Tviol, accuracy]
            accuracy_df['GPT_Verb_sampling'] = score_values




            # take the target Verb to retrieve (base form of the verb)
            targetObj = row['DObj_to_retrieve']

            #if the direct object is not present in the original sentence, we set 'nan'
            # so, dObj will be not retrieved
            if targetObj != "nan":

                regexO = rf"\b{targetObj}\b"



                acceptability_cond = dObj_match_condition_sampling(regexO, generated_text1, generated_text2,
                                                                      generated_text3, row['Sentence'])


                # check if that verb is already present in the original sentence.
                # If it is present also in the generated text, there is a double match with the regex
                if acceptability_cond:
                    dObj_accuracy += 1
                    if row['Condition'] == 'T - T':
                        dObj_accuracy_TT += 1
                    elif row['Condition'] == 'T - AT':
                        dObj_accuracy_TAT += 1
                    elif row['Condition'] == 'AT - T':
                        dObj_accuracy_ATT += 1
                    elif row['Condition'] == 'AT - AT':
                        dObj_accuracy_ATAT += 1
                    elif row['Condition'] == 'T - SP violation':
                        dObj_accuracy_Tviol += 1

                    df.loc[index,'Match_dObj_GPTsampling'] = "Yes"

                else:
                    df.loc[index,'Match_dObj_GPTsampling'] = "No"

            else:
                df.loc[index, 'Match_dObj_GPTsampling'] = "No dObj"

            score_values = [dObj_accuracy_TT, dObj_accuracy_TAT, dObj_accuracy_ATT, dObj_accuracy_ATAT, dObj_accuracy_Tviol, dObj_accuracy]
            accuracy_df['GPT_dObj_sampling'] = score_values

    if model == 'GPT' and  sampling == False:

        for index, row in df.iterrows():


            prompt_GPT2 = row['Prompt_GPT']

            # function to create a new sentence based on prompt input. (Text-generation task without sampling technique)
            generated_text = text_generator(prompt_GPT2, max_new_tokens=5, do_sample=False)[0]['generated_text']
            df.loc[index,'Generated_text_GPT_noSampling'] = generated_text



            # take the target Verb to retrieve (base form of the verb)
            targetV = row['Verb_to_retrieve']

            # retrieve the verb stem
            stemmed_V = ps.stem(targetV)

            # use this regex to retrieve every form of the verb, basing on its stem
            # (e.g. 'complete' - stem:complet - 'completing, completed, completes)
            regexV = rf"\b{stemmed_V}[a-z]*\b"


            acceptability1, acceptability2 = Verb_match_condition(regexV, generated_text, row['Sentence'])


            if acceptability1 or acceptability2 :
                accuracy += 1
                if row['Condition'] == 'T - T':
                    accuracy_TT += 1
                elif row['Condition'] == 'T - AT':
                    accuracy_TAT += 1
                elif row['Condition'] == 'AT - T':
                    accuracy_ATT += 1
                elif row['Condition'] == 'AT - AT':
                    accuracy_ATAT += 1
                elif row['Condition'] == 'T - SP violation':
                    accuracy_Tviol += 1

                df.loc[index,'Match_verb_GPT_noSampling'] = "Yes"


            else:
                df.loc[index,'Match_verb_GPT_noSampling'] = "No"


            score_values = [accuracy_TT, accuracy_TAT, accuracy_ATT, accuracy_ATAT, accuracy_Tviol, accuracy]
            accuracy_df['GPT_Verb_noSampling'] = score_values



            # take the target Verb to retrieve (base form of the verb)
            targetObj = row['DObj_to_retrieve']


            if targetObj != "nan":

                regexO = rf"\b{targetObj}\b"


                acceptable = dObj_match_condition(regexO, generated_text, row['Sentence'])

                # check if that verb is already present in the original sentence. If it is present also in the generated text, there is a double match with the regex
                if acceptable:
                    dObj_accuracy += 1
                    if row['Condition'] == 'T - T':
                        dObj_accuracy_TT += 1
                    elif row['Condition'] == 'T - AT':
                        dObj_accuracy_TAT += 1
                    elif row['Condition'] == 'AT - T':
                        dObj_accuracy_ATT += 1
                    elif row['Condition'] == 'AT - AT':
                        dObj_accuracy_ATAT += 1
                    elif row['Condition'] == 'T - SP violation':
                        dObj_accuracy_Tviol += 1

                    df.loc[index,'Match_dObj_noSampling'] = "Yes"

                else:
                    df.loc[index,'Match_dObj_noSampling'] = "No"

            else:
                df.loc[index,'Match_dObj_noSampling'] = "No dObj"


            score_values = [dObj_accuracy_TT, dObj_accuracy_TAT, dObj_accuracy_ATT, dObj_accuracy_ATAT,
                            dObj_accuracy_Tviol, dObj_accuracy]
            accuracy_df['GPT_dObj_noSampling'] = score_values

    if model == 'BERT' and sampling == False:

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

            #token generated by BERT as the most probable filler for that verb position in prompt
            generated_token = fill_mask_BERT(df, prompt_BERT)

            df.loc[index,'Generated_token_BERT'] = generated_token


            # check if that verb is already present in the original sentence.
            # If it is present also in the generated text, there is a double match with the regex
            if re.findall(regex, generated_token, re.I):
                accuracy += 1
                if row['Condition'] == 'T - T':
                    accuracy_TT += 1
                elif row['Condition'] == 'T - AT':
                    accuracy_TAT += 1
                elif row['Condition'] == 'AT - T':
                    accuracy_ATT += 1
                elif row['Condition'] == 'AT - AT':
                    accuracy_ATAT += 1
                elif row['Condition'] == 'T - SP violation':
                    accuracy_Tviol += 1

                df.loc[index,'Match_verb_BERT'] = "Yes"

            else:
                df.loc[index,'Match_verb_BERT'] = "No"

        score_values = [accuracy_TT, accuracy_TAT, accuracy_ATT, accuracy_ATAT, accuracy_Tviol, accuracy]
        accuracy_df['BERT'] = score_values





sampling = True
score(prompt_df, 'GPT', sampling)
print()
print("GPT text generation with sampling has been done!")
print()

sampling = False
score(prompt_df, 'GPT', sampling)
print()
print("GPT text generation without sampling has been done!")
print()
score(prompt_df, 'BERT', sampling)
print()
print("BERT fill-mask task has been done!")
print()


prompt_df.to_csv(
        '/Users/caput/ELLie-ellipsis_and_thematic_fit_with_TLMs/outputs_csv/Prompts_with_results.csv')
print("Dataset with prompts correctly updated!")



accuracy_df.to_csv(
        '/Users/caput/ELLie-ellipsis_and_thematic_fit_with_TLMs/outputs_csv/Verb_dObj_retrieval_accuracy.csv')
print("Dataset with accuracy scores correctly saved!")