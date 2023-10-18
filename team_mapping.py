# -*- coding: utf-8 -*-

import pickle

import data_manager as dm


bw_team_dict = {'76ers': ['bwPhi', 'bwSixers'],
                'Bucks': ['bwMil'],
                'Bulls': ['bwChi'],
                'Cavaliers': ['bwCavs', 'bwCle'],
                'Celtics': ['bwBos'],
                'Clippers': ['bwLac'],
                'Grizzlies': ['bwMem'],
                'Hawks': ['bwAtl'],
                'Heat': ['bwMia'],
                'Hornets': ['bwCha'],
                'Jazz': ['bwUta'],
                'Knicks': ['bwNyk'],
                'Lakers': ['bwLal'],
                'Mavericks': ['bwDal', 'bwMavs'],
                'Nets': ['bwBkn'],
                'Nuggets': ['bwDen', 'bwNuggets'],
                'Pacers': ['bwInd'],
                'Pelicans': ['bwNol'],
                'Raptors': ['bwTor'],
                'Spurs': ['bwSas'],
                'Suns': ['bwPhx'],
                'Timberwolves': ['bwMin', 'bwWolves'],
                'TrailBlazers': ['bwBlazers', 'bwPor'],
                'Warriors': ['bwGsw'],
                'Wizards': ['bwWas']}

country_flairs = ['ARG', 'AUS', 'BRA', 'CAN', 'CHN', 'CZE', 'DEU', 'DOM',
                  'ESP', 'FRA', 'GFL', 'GRD', 'GRE', 'IRN', 'ITA', 'JOR',
                  'JPN', 'KOR', 'LTU', 'MON', 'NGR', 'NZL', 'PHI', 'PRT',
                  'SEN', 'SLV', 'SRB', 'TUN', 'TUR', 'USA', 'USA1', 'YAC',
                  'Australia', 'Australia1', 'Bobcats3', 'Canada', 'Finland',
                  'GreatBritain', 'Greece', 'Mexico', 'NewZealand',
                  'Philippines', 'Serbia', 'Slovenia', 'SouthKorea', 'Tunisia',
                  'Turkey']


def get_all_flairs():
    all_flairs = set()

    for i in range(1, 5):
        print('round: ', i)
        df_comments = dm.loadNbaDatasetPerRound(i)
        for flair in df_comments['author_flair_css_class'].unique():
            if flair:
                all_flairs.add(flair)

    return all_flairs


def is_team_in_flair(team, flair):
    other_names = {
        'Mavericks': ['Mavs'],
        'Hornets': ['Bobcats'],
        'Cavaliers': ['Cavs'],
        'Wizards': ['Bullets'],
        'Knicks': ['KnickerBockers'],
        'NBA': ['West', 'East']
    }
    if team.lower() in flair.lower():
        return True

    if team in other_names:
        for name in other_names[team]:
            if name.lower() in flair.lower():
                return True
    return False


def create_flair_mapping():
    """
    Return a dict with a flair string key and a team string value.
    """
    all_flairs = get_all_flairs()

    team_flairs = ['76ers', 'Bucks', 'Bulls', 'Cavaliers', 'Celtics',
                   'Clippers', 'Grizzlies', 'Hawks', 'Heat', 'Hornets', 'Jazz',
                   'Kings', 'Knicks', 'Lakers', 'Magic', 'Mavericks', 'Nets',
                   'Nuggets', 'Pacers', 'Pelicans', 'Pelicans', 'Pistons',
                   'Raptors', 'Rockets', 'Spurs', 'Suns', 'SuperSonics',
                   'Thunder', 'Timberwolves', 'TrailBlazers', 'Warriors',
                   'Wizards', 'NBA', 'Country']

    bw_to_team = dict()
    for key in bw_team_dict:
        for bw_flair in bw_team_dict[key]:
            bw_to_team[bw_flair] = key

    country_to_team = {flair: 'Country' for flair in country_flairs}

    flair_to_nba_team = dict()
    for flair in all_flairs:
        for team in team_flairs:
            if is_team_in_flair(team, flair):
                flair_to_nba_team[flair] = team

    flair_to_team = {**flair_to_nba_team, **country_to_team,
                     **bw_to_team, }

    exception_flairs = ['Braves', 'VOTE', 'TorHuskies', 'SUP8',
                        'Generals', 'none']
    for flair in all_flairs:
        if flair not in flair_to_team:
            assert(flair in exception_flairs)

    return flair_to_team


def pickle_flair_to_team(flair_to_team, dir_path=None):
    if dir_path is None:
        dir_path = '/Users/ignaciomoreno/Desktop/NBA project/'

    save_path = dir_path + 'flair_to_team_dict.pickle'
    with open(save_path, 'wb') as handle:
        pickle.dump(flair_to_team, handle)


def create_flair(df, flair_to_team):
    df['team_flair'] = df.author_flair_css_class.apply(
        lambda x: flair_to_team[x] if x in flair_to_team else 'None')

    return df
