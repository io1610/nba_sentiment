# -*- coding: utf-8 -*-

import pandas as pd
import re
import time
from tqdm import tqdm

import data_manager as dm
import pronoun_rename
import process_text
import sentiment_analysis
import utils


def create_you_coref_text(data_manager=None):
    tqdm.pandas()

    if data_manager is None:
        data_manager = dm.DataManager()

    print('Getting all comments...')
    cmts_all = list()
    for round_num in range(1, 5):
        df_cmts = data_manager.load_nba_data_per_round(round_num)
        df_cmts = df_cmts[['name', 'body']].copy()
        cmts_all.append(df_cmts)

    df_all_cmts = pd.concat(cmts_all)
    df_all_cmts.drop_duplicates(subset='name', inplace=True)

    # replace you with you are
    df_all_cmts['body_you'] = df_all_cmts['body'].progress_apply(
        lambda x: re.sub(
            r'\byou\'re\b', 'you are', x, flags=re.IGNORECASE)
    )

    print(process_text)
    df_all_cmts['has_you'] = process_text.has_entity_df(
        df_all_cmts, ['you', 'yourself', 'your', "you're"],
        is_ignorecase=True, text_col='body')

    df_text = df_all_cmts[df_all_cmts['has_you']][['name', 'body_you']].copy()

    df_text['body_you'] = df_text['body_you'].progress_apply(
        lambda x: process_text.clean_text(x))
    df_text = df_text[df_text['body_you'].apply(
        lambda x: (len(x.split()) <= 384))]
    df_text.reset_index(drop=True, inplace=True)

    return df_text


def split_and_save_you_text(data_manager, max_num_comments=100000):
    df_text_you = create_you_coref_text(data_manager)

    indexes = utils.split_with_numpy(df_text_you.index, max_num_comments)
    for num, index in enumerate(indexes):
        df_text_you_sub = df_text_you.loc[index]
        print(df_text_you_sub.shape)

        data_manager.save_coreference_text(df_text_you_sub, round_type='you',
                                           split_num=num,
                                           data_type='preprocessed')


# Run Coreference Resolution

def create_text(texts):
    new_texts = list()
    for text in texts:
        new_text = f"Alex responded to Emma, \"{text}\""
        new_texts.append(new_text)

    return new_texts


def get_processed_text(doc):
    start_str = "Alex responded to Emma, \""
    new_text = doc._.pron_resolved_text
    return new_text[len(start_str):-1]


def run_fastcore_pronoun_youActual(texts, interval):
    you_pronouns = {'you', 'yourself', 'your'}
    resolvable_words = ['Emma']
    nlp = pronoun_rename.get_nlp_model(you_pronouns, resolvable_words)

    new_texts = list()
    tl_start_time = time.time()
    for i in range(0, len(texts), interval):
        start_time = time.time()
        tmp_texts = texts[i:i+interval]
        docs = list(nlp.pipe(tmp_texts, batch_size=256*6))
        # all_docs += docs

        new_texts += [get_processed_text(doc) for doc in docs]

        print(f"interval: {i//interval} time: {time.time() - start_time:.2f}")

    print(f"Total time: {time.time() - tl_start_time:.2f}")
    return new_texts


def run_coref_pronoun_you(data_manager=None, interval=256*4):
    if data_manager is None:
        data_manager = dm.DataManager()

    split_nums = data_manager.num_text_splits(round_type='you',
                                              data_type='preprocessed')
    for split_num in range(split_nums):
        df_threads = data_manager.load_coreference_text(
            round_type='you', split_num=split_num, data_type='preprocessed')

        texts = create_text(df_threads['body_you'].to_list())

        df_threads['processed_text'] = run_fastcore_pronoun_youActual(
            texts, interval)

        data_manager.save_coreference_text(
            df_threads, round_type='you', split_num=split_num,
            data_type='processed')

    return


# Run Sentiment Analysis

def create_you_aspect_text(data_manager=None):
    """
    # Return the dataframe to score all the you pronouns. Note 'You' pronouns
    have been replaced by Emma. "Emma" has next to no overlap in the text
    corpus we are using.
    """
    if data_manager is None:
        data_manager = dm.DataManager()

    split_nums = data_manager.num_text_splits(round_type='you',
                                              data_type='processed')

    data = list()
    for split_num in range(split_nums):
        df_processed_text = data_manager.load_coreference_text(
            round_type='you', split_num=split_num, data_type='processed')

        data.append(df_processed_text)
    df_text = pd.concat(data)

    print(df_text.shape)
    df_text = df_text[df_text['processed_text'].apply(lambda x: 'Emma' in x)]
    df_text.reset_index(drop=True, inplace=True)
    df_text = df_text.copy()
    df_text['aspect'] = 'Emma'
    print(df_text.shape)

    return df_text


def get_sentiment_emma(data_manager):
    df_text = create_you_aspect_text(data_manager)
    aspects = ['Emma']
    df_scores = sentiment_analysis.score_aspect_sentiment(
        df_text, aspects, text_col='processed_text')

    data_manager.save_sentiment(df_scores, round_type='you',
                                team='nba', data_type='sent_score')
    return
