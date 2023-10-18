# -*- coding: utf-8 -*-

import pandas as pd

from collections import defaultdict

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from tqdm import tqdm
from pynvml import *

import numpy as np
import re
import torch
import time

import data_manager as dm
import entity_generator
import process_text
import utils


def replace_entity_words(df, team_entities, entity_key="Heat",
                         show_progress=False):
    word_entities = entity_generator.get_words(team_entities, entity_key)
    pattern = r'\b%s\b' % r'\b|\b'.join(map(re.escape, word_entities))

    if team_entities[entity_key]['lower_case']:
        big_regex = re.compile(pattern, re.IGNORECASE)
    else:
        big_regex = re.compile(pattern)

    if show_progress:
        df['body3'] = df['body3'].progress_apply(
            lambda x: big_regex.sub(entity_key, x))
    else:
        df['body3'] = df['body3'].apply(
            lambda x: big_regex.sub(entity_key, x))

    return df


def create_df_has_entity(df, team_entities, text_col='body2_clean'):
    words_case_sensitive = entity_generator.get_all_words(team_entities,
                                                          case='lower')

    # Grab words that are case insensitive. This will be the case upper words
    # with 'lower_case' = True.
    words_ignore_case = entity_generator.get_all_words(team_entities,
                                                       case='upper')

    has_ignore_case = process_text.has_entity_df(df, words_ignore_case,
                                                 True, text_col)
    has_case_sensitive = process_text.has_entity_df(df, words_case_sensitive,
                                                    False, text_col)

    df['has_entity'] = (has_ignore_case | has_case_sensitive)

    return df


def get_aspects(team, data_manager):
    """
    Return all the aspects from a given team.
    """
    team_entities = entity_generator.get_team_entities([team], data_manager)
    aspects = list(team_entities.keys())
    return aspects


def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used


def empty_gpu_cache(device, MAX_ALLOWED_MEM):
    if "cuda" not in device:
        return

    if get_gpu_utilization() > MAX_ALLOWED_MEM:
        torch.cuda.empty_cache()
        time.sleep(4)


def create_score_df(idds, results, aspect):
    """
    Convert the results list into a DataFrame containing all the scores.
    """
    aspect = aspect.lower()
    col_names_dict = {"Negative": f'{aspect}_neg',
                      "Neutral": f'{aspect}_neu',
                      "Positive": f'{aspect}_pos'}

    data = defaultdict(list)
    data['name'] = idds

    for result in results:
        for score_dict in result:
            label = score_dict['label']
            score = score_dict['score']
            key = col_names_dict[label]
            data[key].append(score)

    df = pd.DataFrame(data)

    # Make the DataFrame sparse
    for col in col_names_dict.values():
        df[col] = df[col].astype(pd.SparseDtype(np.float64))
    return df


def get_data(dataset):
    for data in dataset:
        yield data


def score_aspect_sentiment(df_text, aspects, text_col='body3',
                            score_path=None):
    tqdm.pandas()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = "yangheng/deberta-v3-base-absa-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name).to(device)

    classifier = pipeline("text-classification",
                          model=model, tokenizer=tokenizer,
                          #max_seq_len= 300,
                          device=device
                          )

    df_all_scores = df_text[['name']]

    MAX_ALLOWED_MEM = int(11.5 * 2**30)
    batch_size = 16
    with torch.no_grad():
        for aspect in aspects:
            print(aspect)
            df_sub_text = df_text[df_text['aspect'] == aspect]
            if len(df_sub_text) == 0:
                print(f"aspect {aspect} is skipped!")
                continue

            idds = df_sub_text['name'].values
            texts = df_sub_text[text_col].values
            results = list()
            i = 0
            for res in tqdm(classifier(get_data(texts),
                                       text_pair=aspect,
                                       top_k=None, batch_size=batch_size),
                            total=len(texts)):
                results += [res]

                if i % 50 == 0:
                    empty_gpu_cache(device, MAX_ALLOWED_MEM)
                i += 1

            df_scores = create_score_df(idds, results, aspect)
            df_all_scores = pd.merge(df_all_scores, df_scores, how='left')
            empty_gpu_cache(device, MAX_ALLOWED_MEM)

    return df_all_scores


def create_sentiment_text(round_num, data_manager=None, max_token_len=384):
    """
    Returns a DataFrame containing the text that will be scored by the
    sentiment analysis model.

    Filters the proccessed text with fixed pronouns to only include rows with
    entities for a given playoff round. # save load
    """
    if data_manager is None:
        data_manager = dm.DataManager()

    tqdm.pandas()

    teams_round = utils.get_team_rounds(round_num)

    df_cmts = data_manager.load_cmt_with_processed_text(round_num)
    df_cmts = df_cmts[~df_cmts['body2'].apply(lambda x: isinstance(x, float))]
    df_cmts.reset_index(drop=True, inplace=True)

    # Clean text by removing urls.
    df_cmts['body2_clean'] = df_cmts['body2'].progress_apply(
        lambda x: process_text.clean_text(x))

    team_entities = entity_generator.get_team_entities(teams_round,
                                                       data_manager)

    print("Checking if text contains an entity...")
    df_cmts = create_df_has_entity(df_cmts, team_entities,
                                   text_col='body2_clean')

    df_cmts2 = df_cmts[df_cmts['has_entity']]
    df_cmts2 = df_cmts2[df_cmts2['body2_clean'].apply(
        lambda x: (len(x.split()) <= max_token_len))]

    df_cmts2.reset_index(drop=True, inplace=True)
    df_cmts2['body3'] = df_cmts2['body2_clean']

    cols = ['name', 'meta.timestamp', 'body2_clean']
    df_text = df_cmts2[cols].copy()

    print(f"Full data {len(df_cmts)}; Filtered data {len(df_text)}")
    return df_text


def split_and_save_sent_text_by_team(df_text, round_num, data_manager):
    """
    Splits and saves the sentiment text by team. This function creates and
    saves a DataFrame for each team which includes only comments where an
    entity from the team is mentioned. The teams are selected depending on the
    round of the playoffs.
    """

    teams_round = utils.get_team_rounds(round_num)

    for team in teams_round:
        team_entities = entity_generator.get_team_entities([team])

        aspects = list(team_entities.keys())

        aspect_texts = list()
        tl_start_time = time.time()
        for aspect in aspects:
            df_text['body3'] = df_text['body2_clean']
            df_text = replace_entity_words(df_text, team_entities, aspect,
                                           show_progress=False)

            df_sub_text = df_text[df_text['body3'].apply(
                lambda x: (aspect in x) and (len(x.split()) <= 384))]
            df_sub_text.reset_index(drop=True, inplace=True)
            df_sub_text = df_sub_text.copy()
            df_sub_text['aspect'] = aspect

            aspect_texts.append(df_sub_text[['name', 'body3', 'aspect']])

        df_team_text = pd.concat(aspect_texts)
        print(f'{team} text: {df_team_text.shape[0]} '
              f'total time: {time.time() - tl_start_time:.2f}')

        data_manager.save_sentiment(df_team_text, round_num, team)
