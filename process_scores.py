# -*- coding: utf-8 -*-

import datetime
import pandas as pd
import numpy as np

import nba_schedule
import team_mapping

col_cmts = ['name', 'link_id', 'permalink', 'body', 'parent_id',
            'team_flair', 'score', 'author_flair_css_class',
            'meta.timestamp', 'author']


def get_max_round(df_sch, team):
    return df_sch[df_sch['team'] == team]['round'].max()


def get_all_comments(data_manager, round_num=4):
    """
    Return a DataFrame containing all the text data.
    """
    flair_to_team = data_manager.load_pickle_flair_to_team()

    comment_data = list()
    for i in range(1, round_num+1):
        df_comments = data_manager.load_nba_data_per_round(i)
        df_comments = team_mapping.create_flair(df_comments, flair_to_team)

        comment_data.append(df_comments[col_cmts])

    df_all_comments = pd.concat(comment_data)
    df_all_comments.drop_duplicates(inplace=True)
    return df_all_comments


def aggregate_scores_for_team(data_manager, team, df_all_comments):
    """
    Return the DataFrame containing the sentiment scores for a given team.
    """
    col_cmts = ['name', 'link_id', 'permalink', 'body', 'parent_id',
                'team_flair', 'score', 'author_flair_css_class',
                'meta.timestamp', 'author']

    df_sch = nba_schedule.getSchedule(2022)
    max_round = get_max_round(df_sch, team)

    seen_idds = set()
    all_data = list()
    for num in range(1, max_round+1):
        df_sent = data_manager.load_sentiment(num, team,
                                              data_type='sent_score')

        # remove duplicates
        cnt = df_sent.shape[0]
        df_sent = df_sent[~df_sent.duplicated()]
        df_sent.reset_index(drop=True, inplace=True)
        print(f"Removed {100 - 100 *len(df_sent)/cnt:.2f}% ({cnt}) duplicates")

        # Filter duplicates
        cnt = df_sent.shape[0]
        df_sent = df_sent[~df_sent['name'].isin(seen_idds)]
        df_sent.reset_index(drop=True, inplace=True)
        print(
            f"Removed {100 - 100*len(df_sent)/cnt:.2f}% ({cnt}) seen idds...")
        print(f"After {len(df_sent)}")

        seen_idds.update(df_sent.name.values)

        print("Get all the data...")
        df_data = pd.merge(df_all_comments[col_cmts], df_sent, on='name')
        print(f"Data before: {df_sent.shape} Data after: {df_data.shape}")
        all_data.append(df_data)

    cols_all = set.intersection(*[set(data.columns) for data in all_data])
    cols = [c for c in all_data[0].columns if c in cols_all]
    all_data2 = [data[cols] for data in all_data]
    df_scores = pd.concat(all_data2, axis=0)

    return df_scores, cols


def first_and_last_date(df_sch, team):
    min_date = df_sch[df_sch['team'] == team].date.min()
    max_date = df_sch[df_sch['team'] == team].date.max()

    return max_date, min_date


def find_opps_eliminated(df_sch, team):
    max_round = get_max_round(df_sch, team)

    opps, eli_dates = list(), list()
    for i in range(1, max_round):
        df_tmp = df_sch[(df_sch['team'] == team) & (df_sch['round'] == i)]
        eli_date = df_tmp.date.max() + datetime.timedelta(hours=2, minutes=30)
        opp = df_tmp['Opp'].unique()[0].split()[-1]
        opps.append(opp)
        eli_dates.append(eli_date)

    return opps, eli_dates


def get_sentiment_scores(data_manager, teams, df_all_comments):
    """
    Return a dictionary containing the sentiment scores for every team. The
    key is the name of the team and the value is the DataFrame containing the
    data.
    """
    sentiment_dict = dict()

    for team in teams:
        print(team)
        df_team, team_cols = aggregate_scores_for_team(data_manager,
                                                       team, df_all_comments)
        sentiment_dict[team] = {'df_team': df_team, 'team_cols': team_cols}

    return sentiment_dict


def aggregate_scores_all(df, cols):
    return np.nanmean(df[cols].to_numpy())


def all_scores_get_dataframe(sentiment_dict):
    score_data = list()
    for team in sentiment_dict.keys():
        print(team)
        df_sent = sentiment_dict[team]['df_team']
        sent_cols = sentiment_dict[team]['team_cols']
        neg_cols = [c for c in sent_cols if 'neg' in c]

        df_sent['sum_neg'] = df_sent[neg_cols].mean(axis=1)

        df_sent['neg_to'] = team
        col = ['name', 'team_flair', 'sum_neg', 'neg_to', 'meta.timestamp']
        score_data.append(df_sent[col])

    df_all_scores = pd.concat(score_data)

    return df_all_scores
