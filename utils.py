# -*- coding: utf-8 -*-

import math

import pandas as pd
import numpy as np
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def memory_usage_df(df):
    """
    Return the memory usage of a DataFrame.
    """
    if df.shape[1] == 1:
        return convert_size(df.memory_usage())
    return convert_size(df.memory_usage().sum())


def get_players_df_season(season='2021-22'):
    """
    Return a DataFrame containing all the players for a given season. This
    function uses the NBA's API>
    """
    nba_teams = teams.get_teams()

    # Get the roster for all teams in the season.
    all_players = list()
    for team_info in nba_teams:
        team_id = team_info['id']
        team_name = team_info['full_name']
        roster = commonteamroster.CommonTeamRoster(
            team_id=team_id, season='2021-22')
        roster_data = roster.get_data_frames()[0]
        roster_data['full_team'] = team_name
        all_players.append(roster_data)

    df_players = pd.concat(all_players, axis=0)
    df_players['team'] = df_players['full_team'].apply(lambda x: x.split()[-1])
    return df_players


def save_df_player(df_players):
    path = '/Users/ignaciomoreno/Desktop/NBA project/df_nba_players.csv'
    df_players.to_csv(path)


def get_team_rounds(round_num):
    teams_per_round = {
        4: ['Warriors', 'Celtics',],
        3: ['Mavericks', 'Heat', ],
        2: ['Suns', 'Grizzlies', '76ers', 'Bucks',],
        1: ['Pelicans', 'Jazz', 'Nuggets', 'Timberwolves',
            'Hawks', 'Raptors', 'Bulls', 'Nets']
    }

    teams_round = [team for i in range(round_num, 5)
                   for team in teams_per_round[i]]
    return teams_round


def split_with_numpy(numbers, chunk_size):
    indices = np.arange(chunk_size, len(numbers), chunk_size)
    return np.array_split(numbers, indices)  # [0].tolist()
