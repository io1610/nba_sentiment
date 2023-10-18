# -*- coding: utf-8 -*-

import os

import datetime
import dir_config_manager
import pandas as pd
import pickle
import numpy as np

name_to_abb = {
 'nba': 'NBA',
 'Hawks': 'ATL',
 'Celtics': 'BOS',
 'Cavaliers': 'CLE',
 'Pelicans': 'NOP',
 'Bulls': 'CHI',
 'Mavericks': 'DAL',
 'Nuggets': 'DEN',
 'Warriors': 'GSW',
 'Rockets': 'HOU',
 'Clippers': 'LAC',
 'Lakers': 'LAL',
 'Heat': 'MIA',
 'Bucks': 'MIL',
 'Timberwolves': 'MIN',
 'Nets': 'BKN',
 'Knicks': 'NYK',
 'Magic': 'ORL',
 'Pacers': 'IND',
 '76ers': 'PHI',
 'Suns': 'PHX',
 'Blazers': 'POR',
 'Kings': 'SAC',
 'Spurs': 'SAS',
 'Thunder': 'OKC',
 'Raptors': 'TOR',
 'Jazz': 'UTA',
 'Grizzlies': 'MEM',
 'Wizards': 'WAS',
 'Pistons': 'DET',
 'Hornets': 'CHA'}


def getNewTimeStamp(df, time_col='timestamp'):
    def createDate(x):
        if np.isnan(x):
            return None
        return datetime.datetime.utcfromtimestamp(
            int(x))

    df['meta.timestamp'] = df[time_col].apply(lambda x: createDate(x))
    return df


default_dir_config = dir_config_manager.default_dir_config


class DataManager():
    rnum_to_name = {1: 'first', 2: 'second', 3: 'third', 4: "fourth"}

    def _create_dir_if_not_exists(self, dir_path):
        isExist = os.path.exists(dir_path)
        if not isExist:
            os.makedirs(dir_path)

    def __init__(self, dir_config=None):
        if dir_config is None:
            dir_config = default_dir_config
        self.data_dir = dir_config['data_dir']
        self.processed_dir = dir_config['processed_dir']
        self.pre_processed_dir = dir_config['pre_processed_dir']
        self.score_dir = dir_config['score_dir']
        self.sent_text_dir = self.data_dir + 'sent_text/'

        self._create_dir_if_not_exists(self.processed_dir)
        self._create_dir_if_not_exists(self.score_dir)
        self._create_dir_if_not_exists(self.sent_text_dir)

        # self.dir_colab = self.data_dir + 'colab_data/'

    def _retrieve_file_and_dir(self, data_type, round_type=None,
                               split_num=None, team=None):
        if data_type == 'preprocessed':
            dir_path = self.pre_processed_dir + f"pronoun_text_{round_type}/"
            file_name = f'df_pretext_{round_type}_{split_num}.parquet.gzip'

        elif data_type == 'processed':
            dir_path = self.processed_dir + f"pronoun_text_{round_type}/"
            file_name = f'df_text_{round_type}_{split_num}.parquet.gzip'

        elif data_type == 'sent_text':
            abbr = name_to_abb[team].lower()
            dir_path = self.sent_text_dir + f"round_{round_type}/"
            file_name = f'df_text_to_score_{abbr}_{round_type}.gzip'
        elif data_type == 'sent_score':
            abbr = name_to_abb[team].lower()
            dir_path = self.score_dir + f"round_{round_type}/"
            file_name = f"df_all_scores_{round_type}_{abbr}.csv"

        return dir_path, file_name

    def save_coreference_text(self, df_text, round_type, split_num,
                              data_type='preprocessed'):
        dir_path, file_name = self._retrieve_file_and_dir(
            data_type, round_type, split_num)
        self._create_dir_if_not_exists(dir_path)
        df_text.to_parquet(dir_path + file_name, compression='gzip')

    def load_coreference_text(self, round_type, split_num,
                              data_type='preprocessed'):
        dir_path, file_name = self._retrieve_file_and_dir(
            data_type, round_type, split_num)
        df_text = pd.read_parquet(dir_path + file_name)
        return df_text

    def save_sentiment(self, df_sent, round_type, team,
                       data_type='sent_text'):
        dir_path, file_name = self._retrieve_file_and_dir(
            data_type, round_type, team=team)
        self._create_dir_if_not_exists(dir_path)
        df_sent.to_csv(dir_path + file_name, index=False, compression='gzip')

    def load_sentiment(self, round_type, team, data_type='sent_text'):
        dir_path, file_name = self._retrieve_file_and_dir(
            data_type, round_type, team=team)
        self._create_dir_if_not_exists(dir_path)
        df_sent = pd.read_csv(dir_path + file_name, lineterminator='\n',
                              compression='gzip')
        return df_sent

    def num_text_splits(self, round_type, data_type='processed'):
        dir_path, file_name = self._retrieve_file_and_dir(
            data_type, round_type, split_num=0)
        dir_list = os.listdir(dir_path)
        return len(dir_list)

    def load_nba_data_per_round(self, round_num, is_comment=True):
        """
        Returns Nba comment dataset for a given playoff round.
        """
        round_name = self.rnum_to_name[round_num]
        if is_comment:
            file = f'df_nba_comment_{round_name}_round.parquet.gzip'
        else:
            file = f'df_nba_submi_{round_name}_round.parquet.gzip'

        df = pd.read_parquet(self.data_dir + file)
        df = getNewTimeStamp(df, 'created_utc')

        return df

    def load_all_data_per_round(self, round_num):
        """
        Load comment and submission data for a given playoff round.
        """
        df_comments = self.load_nba_data_per_round(round_num)
        df_submi = self.load_nba_data_per_round(round_num, is_comment=False)

        submi_link_ids = set(df_submi['name'].unique())
        df_comments = df_comments.loc[df_comments['link_id'].isin(
            submi_link_ids), :]
        df_comments.reset_index(drop=True, inplace=True)

        return df_comments, df_submi

    def load_split_data(self, round_num, split_num):
        """
        Return DataFrame containing splited comments text data and submission
        DataFrame.
        """
        df_submi = self.load_nba_data_per_round(round_num, is_comment=False)

        df_comments = self.load_coreference_text(round_num, split_num,
                                                 data_type='preprocessed')

        link_ids = set(df_comments.link_id.unique())
        df_submi = df_submi.loc[df_submi['name'].isin(link_ids), :].reset_index(
            drop=True)

        return df_comments, df_submi

    def load_df_player(self):
        """
        Return the DataFrame containing all nba players.
        """
        path = self.data_dir + 'df_nba_players.csv'
        df_players = pd.read_csv(path, index_col=0)
        return df_players

    def load_processed_text(self, round_num):
        """
        Load the processed data with resolved pronouns.
        """
        split_nums = self.num_text_splits(round_num, data_type='processed')
        processed_cmts = list()
        for split_num in range(split_nums):
            df_tmp = self.load_coreference_text(round_num, split_num,
                                                data_type='processed')
            processed_cmts.append(df_tmp)

        df_processed_cmts = pd.concat(processed_cmts)

        return df_processed_cmts

    def load_cmt_with_processed_text(self, round_num):
        df_processed_cmts = self.load_processed_text(round_num)
        df_comments = self.load_nba_data_per_round(round_num, is_comment=True)

        cols = ['name', 'has_pronoun', 'body2', 'pseudonym']
        df_cmts = pd.merge(df_comments, df_processed_cmts[cols], on='name')
        return df_cmts

    def load_pickle_flair_to_team(self, dir_path=None):
        load_path = self.data_dir + 'flair_to_team_dict.pickle'
        with open(load_path, 'rb') as f:
            flair_to_team = pickle.load(f)

        return flair_to_team

    def load_names(self):
        load_path = self.data_dir + 'lotr_characters_data2.csv'
        df_names = pd.read_csv(load_path)
        return df_names
