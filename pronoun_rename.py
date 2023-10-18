# -*- coding: utf-8 -*-

from collections import defaultdict
import time

from itertools import combinations_with_replacement
import pandas as pd
import spacy
import string
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

import process_text
import spacy_resolve_pronoun
import utils

# resolvable_pronouns
resolvable_pronouns = {'he', 'her', 'herself', 'him', 'himself', 'she', 'his',
                       'it', 'its', 'itself', 'mine', 'my', 'myself', 'our',
                       'ours', 'ourselves',  'them', 'themself', 'themselves',
                       'they', 'us', 'we', 'you', 'your', 'yours', 'yourself',
                       'yourselves'}

# These are pronouns that we want to resolve. The missing pronouns don't add
# value for sentiment analysis.
# Missing: ['i', 'mine', 'my', 'myself', 'you', 'your', 'their', 'theirs',
# 'themselves',]
pronouns_to_process = {'he', 'her', 'herself', 'him', 'himself', 'she', 'his',
                       'it', 'its', 'itself',  'our', 'ours', 'ourselves',
                       'them', 'themself', 'they', 'us', 'we', 'yours',
                       'yourself', 'yourselves'}


class ConversationManager:
    """
    A text managing class designed for preparing input data for the model
    including if a comment has a pronoun.
    """

    def __init__(self, df_threads, link_id_to_title, df_names):
        self.link_id_to_title = link_id_to_title
        self.id_to_user = self.create_idd_to_user(df_threads, df_names)
        self.id_has_pronoun = df_threads.set_index(
            'name')['has_pronoun'].to_dict()
        self.id_to_link = df_threads.set_index('name')['link_id'].to_dict()

    def prepare_text_for_model(self, cmt_chain, title, id_to_cmt, cmt_to_text):
        """
        Returns a text to feed into the coreference resolution model. We add
        the context of the title and the previous parent comment.
        """
        text = f"\nTitle: {title}\n"

        for i in range(max(0, len(cmt_chain) - 3), len(cmt_chain)):
            if i == 0:
                text += f"{id_to_cmt[cmt_chain[0]]} wrote:\n"
            else:
                text += f"{id_to_cmt[cmt_chain[i]]} responded to "\
                    f"{id_to_cmt[cmt_chain[i-1]]}:\n"
            if i >= len(cmt_chain) - 3:
                text += cmt_to_text[cmt_chain[i]]
                if i != len(cmt_chain) - 1:
                    text += "\n"

        return text

    def name_gen(self, df_names):
        """
        Returns a generator that yields name combinations from df_names.
        """

        names = df_names['name'].tolist()
        lowercase = [''] + list(string.ascii_lowercase)

        for char_tup in combinations_with_replacement(lowercase, 6):
            c = ''.join(char_tup)
            for name in names:
                yield name + c

    def create_idd_to_user(self, df_threads, df_names):
        link_ids = df_threads['link_id'].unique()

        id_to_user = dict()

        for link_id in link_ids:
            gen = self.name_gen(df_names)
            df_thread = df_threads[df_threads['link_id'] == link_id]

            for n, i in enumerate(df_thread['name'].unique()):
                id_to_user[i] = next(gen)
        return id_to_user

    def get_commenter_text(self, cmt_chain):
        if len(cmt_chain) == 1:
            commenter_text = f"{self.id_to_user[cmt_chain[0]]} wrote:\n"
        else:
            commenter_text = f"{self.id_to_user[cmt_chain[-1]]} responded to "\
                f"{self.id_to_user[cmt_chain[-2]]}:\n"

        return commenter_text

    # Given an idd return a title.
    def get_title(self, idd):
        return self.link_id_to_title[self.id_to_link[idd]]

    def generate_text(self, cmt_chain, id_to_text):
        title = self.get_title(cmt_chain[-1])
        return self.prepare_text_for_model(cmt_chain, title,
                                           self.id_to_user, id_to_text)

    def has_pronoun(self, idd):
        return self.id_has_pronoun[idd]


def split_and_save_data(data_manager, round_num,
                        is_clean_text=False, max_num_comments=10000):
    """
    Load comment and submission data and split/save them into smaller files.
    """
    tqdm.pandas()
    df_comments, df_submi = data_manager.load_all_data_per_round(round_num)

    if is_clean_text:
        df_comments['body'] = df_comments['body'].progress_apply(
            lambda x: process_text.clean_text(x))
        df_comments = df_comments[df_comments['body'].apply(
            lambda x: (len(x.split()) <= 384))]
        df_comments.reset_index(drop=True, inplace=True)

    cols = ['name', 'id', 'link_id', 'parent_id', 'body']

    group_link_ids = process_text.get_doc_ids(df_comments,
                                              max_num_comments)

    for n, link_ids in tqdm(enumerate(group_link_ids)):
        df_comments2 = df_comments.loc[df_comments['link_id'].isin(
            link_ids), cols]
        df_comments2.reset_index(drop=True, inplace=True)
        data_manager.save_coreference_text(df_comments2, round_num, n,
                                           data_type='preprocessed')


def run_pronoun_coref(df_comments, df_submi, data_manager,
                      max_num_comments=800, max_num_texts=256*8):
    """
    Run coreference resolution for comments which will replace pronouns with
    the noun they refer too. Ex: "He is great!" -> "Giannis is great!"
    """

    print("Checking if comments contain a pronoun...")
    df_comments['has_pronoun'] = process_text.has_entity_df(
        df_comments, list(pronouns_to_process))

    link_id_to_title = df_submi.set_index('name')['title'].to_dict()
    group_link_ids = process_text.get_doc_ids(df_comments, max_num_comments)

    df_names = data_manager.load_names()

    threads = list()
    for n, link_ids in enumerate(group_link_ids):
        print(f"Running iteration {n} link_ids {len(link_ids)}")
        df_threads = df_comments.loc[
            df_comments['link_id'].isin(link_ids), :].copy()

        conver_gen = ConversationManager(
            df_threads, link_id_to_title, df_names)
        df_processed_threads = run_fastcoref_for_threads(df_threads,
                                                         conver_gen,
                                                         max_num_texts)
        # device = cuda.get_current_device()
        # device.reset()

        threads.append(df_processed_threads)

    return pd.concat(threads, axis=0)


def resolve_fastcoref_doc(doc, cmt_chain, conver_gen):
    commenter_text = conver_gen.get_commenter_text(cmt_chain)

    new_text = doc._.pron_resolved_text
    new_text2 = new_text[new_text.rfind(commenter_text) + len(commenter_text):]
    return new_text2


def process_docs_and_update_text(docs, conver_gen, cmt_group_chain,
                                 id_to_text):
    for cmt_chain, doc in zip(cmt_group_chain, docs):
        id_to_text[cmt_chain[-1]] = resolve_fastcoref_doc(
            doc, cmt_chain, conver_gen)
    return id_to_text


def get_nlp_model(resol_pronouns=None, resol_words=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if "cuda" in device:
        spacy.require_gpu()

    resol_pronouns = resol_pronouns if resol_pronouns else resolvable_pronouns

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("fastcoref", config={'enable_progress_bar': False})
    nlp.add_pipe("resolve_fastcoref_pronoun",
                 config={"resolvable_pronouns":
                         {key: 0 for key in resol_pronouns},
                         "resolvable_words": resol_words
                         })

    return nlp


def get_docs_fastcoref(texts, nlp):
    with autocast():
        start_time = time.time()
        docs = list(nlp.pipe(texts))
        print(f"\n{len(texts)} texts in time: {time.time() - start_time:.2f}")
    nlp = None
    return docs


def run_fastcoref_for_threads(df_threads, conver_gen, max_num_texts):
    """
    Run fastcoref for entire comment threads. Since we are adding context from
    parent comments we first need to resolve comments from a top to bottom
    in the commet tree. To do this is will run a BFS where we resolve all
    comments at the top level first. Afterwards, we go to all comments on the
    second level, and so on.

    Returns: The same df_threads with the resolved text and the pseudonym used
    to simulate the conversation.
    """
    nlp = get_nlp_model()

    df_threads['body2'] = df_threads['body']
    id_to_text = df_threads.set_index('name')['body2'].to_dict()

    graph = defaultdict(list)
    cmt_stack = []
    cols = ['name', 'parent_id', 'link_id']
    for idd, parent_id, link_id in df_threads[cols].values:
        if parent_id == link_id:
            cmt_stack.append([idd])
        else:
            graph[parent_id].append(idd)

    total_st_time = time.time()
    iterations = 1
    while cmt_stack:
        if iterations >= 20:
            break

        cur_cmt_stack = [cmt_chain for cmt_chain in cmt_stack
                         if conver_gen.has_pronoun(cmt_chain[-1])]

        print(f'Level {iterations} Len cur_cmt_stack: {len(cur_cmt_stack)}')
        for cmt_group_chain in utils.split_with_numpy(cur_cmt_stack, max_num_texts):
            cmt_group_chain = cmt_group_chain.tolist()
            texts = list()
            for cmt_chain in cmt_group_chain:
                texts.append(conver_gen.generate_text(cmt_chain, id_to_text))

            # Get the text from Scapy
            docs = get_docs_fastcoref(texts, nlp)
            # procss the docs and update the text
            id_to_text = process_docs_and_update_text(docs, conver_gen,
                                                      cmt_group_chain,
                                                      id_to_text)

        new_cmt_stack = list()
        for cmt_chain in cmt_stack:
            for next_id in graph[cmt_chain[-1]]:
                new_cmt_stack.append(cmt_chain + [next_id])
        cmt_stack = new_cmt_stack
        iterations += 1

    print(f"Time: {time.time() - total_st_time}")
    df_threads['body2'] = df_threads['name'].apply(
        lambda x: id_to_text[x])
    df_threads['pseudonym'] = df_threads['name'].apply(
        lambda x: conver_gen.id_to_user[x])
    return df_threads
