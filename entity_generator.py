# -*- coding: utf-8 -*-

import data_manager as dm

remove_name_dict = {
    'Bucks': {'AntetokounmpoT': {'Antetokounmpo'},
              'Nwora': {'Jordan'}},
    'Bulls': {'Jones': {'Derrick'}, 'Vucevic': {'Nikola'}, 'Cook': {'Tyler'},
              'Bradley': {'Tony'}},

    'Celtics': {'WilliamsRo': {'Williams'}, 'Pritchard': {'Payton'},
                'Nesmith': {'Aaron'}, 'Fitts': {'Malik'}, 'Ryan': {'Matt'}},
    'Grizzlies': {'Anderson': {'Kyle'}, 'Clarke': {'Brandon'},
                  'Konchar': {'John'}},

    'Hawks': {'Huerter': {'Kevin'}, 'JohnsonJa': {'Jalen'}, 'Knox': {'Kevin'}},
    'Heat': {'SmartJa': {'Javonte'}},

    'Jazz': {'Gay': {'Rudy'}, 'Sneed': {'Xavier'}, 'Clarkson': {'Jordan'},
             'Forrest': {'Trent'}},

    'Pelicans': {'Clark': {'Gary'}, 'Harper': {'Jared'}, 'Murphy': {'Trey'}},

    'Mavericks': {'WrightMo': {'Moses'}},
    'Nets': {'HarrisJo': {'Joe'}},
    'Nuggets': {'GreenJa': {'Green'}, 'Barton': {'Will'}},

    'Suns': {'Payne': {'Cameron'}, 'HolidayAa': {'Aaron'}},

    'Raptors': {'Boucher': {'Chris'}, 'JohnsonDa': {'David'},
                'Trent': {'Gary'}},


    'Timberwolves': {'Nowell': {'Jaylen'}, 'Okogie': {'Josh'},
                     'Beverley': {'Patrick'}, 'McLaughlin': {'Jordan'}},

    'Warriors': {'Chiozza': {'Chris'}, 'Poole': {'Jordan'},
                 'Wiseman': {'James'}, 'Iguodala': {'Andre'}},
    '76ers': {'Millsap': {'Paul'}, 'Reed': {'Paul'}, 'Springer': {'Jaden'},
              'DeAndre': {'Jordan'}
              },
}

lower_case_dict = {
    'Celtics': [['Smart', 'Smart']],
    'Bucks': [['Holiday', 'Holiday']],
    'Bulls': [['Ball', 'Ball']],
    'Heat': [['Strus', 'Max']],
    'Warriors': [['Green', 'Green']],
}

last_name_dict = {
    'Bucks': ['Tucker'],
    'Bulls': ['Williams', 'White', 'Green', 'Thompson', 'Hill', 'Brown',
              'Thomas'],
    'Celtics': ['Thomas'],

    'Grizzlies': ['Williams', 'Jones'],
    'Hawks': ['Bogdanovic', 'Williams', 'Johnson', 'Brown'],
    'Heat': ['Smart', 'Morris'],

    'Jazz': ['Hernangomez', 'Butler'],

    'Mavericks': ['Brown', 'Green', 'Wright'],
    'Nets': ['Edwards', 'Curry', 'Harris', 'Brown'],
    'Nuggets': ['Green', 'Reed', 'Porter'],

    'Raptors': ['Young', 'Brooks', 'Johnson'],
    'Suns': ['Holiday', 'Payton'],
    '76ers': ['Powell', 'Green', 'Brown'],

}

add_ent_dict = {
    '76ers': [['76ers', 'Philly'],
              ['76ers', 'Sixers'],
              ['76ers', 'Sixer']],
    'Mavericks': [['Mavericks', 'Mavs']],
    'Timberwolves': [['Timberwolves', 'Wolves']],
}

name_exception = {
    'Bucks': {'Thanasis Antetokounmpo': 'AntetokounmpoT'},
    'Celtics': {'Robert Williams III': 'WilliamsRo'},
    'Nuggets': {'JaMychal Green': 'GreenJa'},
    '76ers': {'DeAndre Jordan': 'DeAndre'}
}


def get_words(team_entities, key, make_lower=False):
    list_of_words = list()
    for w in [key] + team_entities[key]['names']:
        list_of_words.append(w)
        if make_lower and team_entities[key]['lower_case']:
            list_of_words.append(w.lower())
    return list_of_words


def get_all_words(team_entities, case='all'):
    """
    Returns a list with all the entities inside team_entities.

    # all: grab all words and make lower if possible
    # upper: grab only upper_case words
    """

    list_of_words = list()
    for key in team_entities:
        if case == 'all':
            list_of_words += get_words(team_entities, key, make_lower=True)
        elif case == 'upper':
            if team_entities[key]['lower_case']:
                list_of_words += get_words(team_entities, key)
        elif case == 'lower':
            if not team_entities[key]['lower_case']:
                list_of_words += get_words(team_entities, key)

    return list_of_words


def remove_suffix_entity(team_entities):
    """
    Remove suffix from entity keys and as individual words.
    """
    suffixes = ['Jr.', 'II', 'III', 'IV']

    for suffix in suffixes:
        if suffix not in team_entities:
            continue

        ent_dict = team_entities[suffix]

        del team_entities[suffix]

        ent_key = ent_dict['names'][0].split()[-2]
        ent_names = ent_dict['names']

        team_entities[ent_key] = {
            'names': [ent for ent in ent_names if ent != suffix],
            'lower_case': ent_dict['lower_case'],
            'team': ent_dict['team']
        }

    return team_entities


def skip_lower_ent(team_entities, team, lower_case_dict):
    """
    Remove the lower case entity. EX: `Max Strus` don't add `max`.
    """
    if team not in lower_case_dict:
        return team_entities

    for key, val in lower_case_dict[team]:
        ent_name = team_entities[key]['names']
        new_ent_name = ent_name
        for name in new_ent_name:
            if name not in val and name != name.lower():
                new_ent_name.append(name.lower())

        team_entities[key]['names'] = new_ent_name
        team_entities[key]['lower_case'] = False

    return team_entities


def remove_last_name(team_entities, team, last_name_dict):
    """
    Remove Last name because another player has the same name.
    Make the key Last name + first charcter.
    """
    if team not in last_name_dict:
        return team_entities

    for last_name in last_name_dict[team]:
        ent_dict = team_entities[last_name]
        del team_entities[last_name]

        ent_key = last_name + ent_dict['names'][0][:2]
        team_entities[ent_key] = {'names': [ent for ent in ent_dict['names']
                                            if ent != last_name],
                                  'lower_case': ent_dict['lower_case'],
                                  'team': ent_dict['team']
                                  }

    return team_entities


def add_ent_name(team_entities, team, add_ent_dict):
    """
    Add extra names that share the same entities. For example adding 76ers to
    Philly.
    """
    if team not in add_ent_dict:
        return team_entities

    for key, val in add_ent_dict[team]:
        team_entities[key]['names'].append(val)

    return team_entities


def remove_ent_name(team_entities, team, remove_name_dict):
    """
    Remove names from entities from team_entities using remove_name_dict. This
    usually involves removing names so there aren't overlaps such as:
        Chris Paul and Paul Millsap.
    """
    if team not in remove_name_dict:
        return team_entities

    for key in remove_name_dict[team]:
        names = team_entities[key]['names']
        set_exclude = remove_name_dict[team][key]
        team_entities[key]['names'] = [name
                                       for name in names
                                       if name not in set_exclude]

    return team_entities


def create_ent_key_and_name(player, team, name_exception, suffixes):
    if team in name_exception and player in name_exception[team]:
        return name_exception[team][player]

    key = player.split()[-1]
    for name in reversed(player.split()):
        if name not in suffixes:
            key = name
            break
    return key


def get_entities_for_team(df_players, team='Heat'):
    team_entities = dict()

    full_team = df_players[df_players['team'] == team]['full_team'].iloc[0]
    team_entities[team] = {'names': [full_team,
                                     ' '.join(full_team.split()[:-1]),
                                     full_team.split()[-1]],
                           'lower_case': True}

    for player in df_players[df_players['team'] == team]['PLAYER'].values:
        suffixes = ['Jr.', 'II', 'III', 'IV']
        key = create_ent_key_and_name(player, team,
                                      name_exception, suffixes)
        names = [name for name in player.split() if name not in suffixes]
        assert (key not in team_entities)

        team_entities[key] = {
            'names': [player] + names,
            'lower_case': True,
            'team': team
        }

    # Clean entities
    ent_exceptions = ['Strus', 'Holiday']
    for ent_exp in ent_exceptions:
        if ent_exp in team_entities:
            team_entities[ent_exp]['lower_case'] = False

    team_entities = remove_suffix_entity(team_entities)
    team_entities = remove_last_name(team_entities, team, last_name_dict)
    team_entities = remove_ent_name(team_entities, team, remove_name_dict)
    team_entities = skip_lower_ent(team_entities, team, lower_case_dict)
    team_entities = add_ent_name(team_entities, team, add_ent_dict)

    return team_entities


def get_team_entities(teams_round, data_manager=None):
    if data_manager is None:
        data_manager = dm.DataManager()

    df_players = data_manager.load_df_player()

    team_entities = dict()
    for team in teams_round:
        team_entities = {**team_entities,
                         **get_entities_for_team(df_players, team)}

    return team_entities
