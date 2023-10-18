# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime
import pytz

import dir_config_manager


def find_start_end_of_eliminated_round(df, team):
    df_tmp = df[(df['team'] == team) & (df['isLastRound'])]
    min_date = df_tmp['date'].min().to_pydatetime()
    max_date = df_tmp['date'].max().to_pydatetime()
    return min_date, max_date


def getDate(df):
    datetime_format = "%a %b %d %Y %I:%M%p %Z"
    tmp = (df['Date'] + " " + df['Start (ET)']) + 'm EST'
    parsed_datetimes = pd.to_datetime(tmp, format=datetime_format)
    datetime_objs = parsed_datetimes.dt.to_pydatetime()
    datetime_objs = [d.astimezone(pytz.utc).replace(tzinfo=None)
                     for d in datetime_objs]

    df['date'] = datetime_objs
    #df['date'] = df['date'].dt.to_pydatetime()
    return df


def getSchedule(year=2022):
    file_name = dir_config_manager.schedule_dict[year]

    df_sch = pd.read_csv(file_name)

    rename_dict = {'Unnamed: 7': 'IsOT'}
    df_sch.rename(columns=rename_dict, inplace=True)

    matchups = list()
    for pair in df_sch[['Visitor/Neutral', 'Home/Neutral']].values:
        pair.sort()
        p = pair[0] + " " + pair[1]
        matchups.append(p)
    df_sch['matchup'] = matchups

    seen_pair = set()
    pair_to_match = dict()
    i = 1
    for matchup in df_sch['matchup'].values:
        if matchup not in seen_pair:
            seen_pair.add(matchup)
            pair_to_match[matchup] = i
            i += 1

    matchup_names = list()
    round_num = list()
    for matchup in df_sch['matchup'].values:
        game_num = pair_to_match[matchup]
        if game_num >= 1 and game_num <= 8:
            matchup_names.append("First Round")
            round_num.append(1)
        elif game_num >= 9 and game_num <= 12:
            matchup_names.append("Conference SemiFinals")
            round_num.append(2)
        elif game_num >= 13 and game_num <= 14:
            matchup_names.append("Conference Finals")
            round_num.append(3)
        else:
            matchup_names.append("Finals")
            round_num.append(4)
    df_sch['matchup_name'] = matchup_names
    df_sch['round'] = round_num

    max_round_reached = df_sch.groupby('Home/Neutral')['round'].max().to_dict()
    max_round_reached['Golden State Warriors'] = 5

    schedule = list()
    teams = df_sch['Home/Neutral'].unique()
    for team in teams:
        df_home = df_sch[df_sch['Home/Neutral'] == team].copy()
        df_home.reset_index(drop=True, inplace=True)
        df_home['isHome'] = True
        rename_home_dict = {'Home/Neutral': 'Team', 'PTS.1': 'PTS',
                            'Visitor/Neutral': 'Opp', 'PTS': 'PTS.1'}

        df_home.rename(columns=rename_home_dict, inplace=True)

        df_vist = df_sch[df_sch['Visitor/Neutral'] == team].copy()
        df_vist.reset_index(drop=True, inplace=True)
        df_vist['isHome'] = False
        rename_vist_dict = {'Home/Neutral': 'Opp',
                            'Visitor/Neutral': 'Team', }
        df_vist.rename(columns=rename_vist_dict, inplace=True)

        schedule.append(df_home)
        schedule.append(df_vist)

    df_sch2 = pd.concat(schedule, axis=0)
    df_sch2.reset_index(drop=True, inplace=True)
    df_sch2['date'] = df_sch2['Date'].apply(lambda x:
                                            datetime.strptime(x,
                                                              "%a %b %d %Y"))
    df_sch2.sort_values(['Team', 'date'], inplace=True)
    df_sch2['Result'] = df_sch2[['PTS', 'PTS.1']].apply(
        lambda x: 'W' if x[0] > x[1] else 'L', axis=1)

    df_sch2['isLastRound'] = [True if max_round_reached[t] == r else False
                              for t, r in df_sch2[['Team', 'round']].values]

    df_sch2['team'] = df_sch2['Team'].apply(lambda x: x.split()[-1])
    df_sch2 = getDate(df_sch2)

    return df_sch2


if __name__ == "__main__":
    df_sch = getSchedule(2022)

    team = 'Phoenix Suns'
    before, after = find_start_end_of_eliminated_round(df_sch, team)
    print(before, after)
