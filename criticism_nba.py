# -*- coding: utf-8 -*-

import datetime
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.formula.api as smf

import data_manager as dm
import process_scores


def get_team_sent_data_and_cols(sentiment_dict, team):
    df_sent_team = sentiment_dict[team]['df_team']
    sent_team_cols = sentiment_dict[team]['team_cols']
    return df_sent_team, sent_team_cols


def first_critcism_by_team(df_sent_twd_user, team, threshold=.8):
    df_neg_from_team = df_sent_twd_user[
        (df_sent_twd_user['neg_from'] == team) &
        (df_sent_twd_user['neg'] >= .8)
    ].copy()

    # Grab the first criticism towards an author.
    df_neg_from_team.sort_values('meta.timestamp', inplace=True)
    df_neg_from_team = df_neg_from_team.groupby('author').first().reset_index()

    # Drop out unneccesary columns
    neg_from_cols = ['author', 'team_flair', 'meta.timestamp']
    df_neg_from_team = df_neg_from_team[neg_from_cols]
    df_neg_from_team.rename(columns={'meta.timestamp': 'cutoff_date'},
                            inplace=True)
    return df_neg_from_team


def split_dataframe_by_time(df, delta_before, delta_after, how='prev'):
    df['time_difference'] = (df['cutoff_date'] - df['meta.timestamp']).abs()

    if how == 'prev':
        df_before = df[df['meta.timestamp'] < df['cutoff_date']]
        df_after = df[df['meta.timestamp'] > df['cutoff_date']]

        # Filter so authors are found before and after.
        authors = set(df_before.author.unique()) & set(
            df_after.author.unique())
        df_before = df_before[df_before['author'].isin(authors)]
        df_after = df_after[df_after['author'].isin(authors)]

        df_before = df_before[(df_before['time_difference']) <= delta_before]
        df_after = df_after[(df_after['time_difference']) <= delta_after]

    else:
        df_before = df[(df['meta.timestamp'] < df['cutoff_date']) &
                       (df['time_difference'] <= delta_before)]
        df_after = df[(df['meta.timestamp'] > df['cutoff_date']) &
                      (df['time_difference'] <= delta_after)]

        # Filter so authors are found before and after.
        authors = set(df_before.author.unique()) & set(
            df_after.author.unique())
        df_before = df_before[df_before['author'].isin(authors)]
        df_after = df_after[df_after['author'].isin(authors)]

    return df_before, df_after


def create_sent_twd_user(df_comments, df_resp_sent):
    """
    Return a dataframe containing a users sentiment toward other user.
    """
    df_resp_sent_add = pd.merge(df_comments[['name', 'parent_id',
                                             'team_flair']],
                                df_resp_sent, on='name')

    pid_to_score = df_resp_sent_add.set_index(
        'parent_id')['emma_neg'].to_dict()
    pid_to_flair = df_resp_sent_add.set_index(
        'parent_id')['team_flair'].to_dict()

    idds = set(pid_to_score.keys())
    df_sent_twd_user = df_comments.loc[df_comments['name'].isin(
        idds), :].copy()

    df_sent_twd_user['neg'] = df_sent_twd_user['name'].apply(
        lambda x: pid_to_score[x] if x in pid_to_score else None)
    df_sent_twd_user['neg_from'] = df_sent_twd_user['name'].apply(
        lambda x: pid_to_flair[x] if x in pid_to_flair else None)

    return df_sent_twd_user


def non_treatment_team_sentiment(team, teams, sentiment_dict,
                                 df_neg_from_team, df_sent_toward_user,
                                 delta_before, delta_after,
                                 team_end_date=None):
    baseline_sent_before, baseline_sent_after = list(), list()
    for team_opp in teams:
        if team_opp == team:
            continue

        df_sent_team_opp, sent_team_cols_opp = get_team_sent_data_and_cols(
            sentiment_dict, team_opp)
        neg_cols_opp = [c for c in sent_team_cols_opp if 'neg' in c]

        df_neg_from_team_with_opp_flair = \
            df_neg_from_team[df_neg_from_team['team_flair'] != team_opp]

        # sentiment team opp subset
        df_sto_subset = pd.merge(df_sent_team_opp,
                                 df_neg_from_team_with_opp_flair, on='author')
        df_sto_subset['sum_neg'] = df_sto_subset[neg_cols_opp].mean(axis=1)

        # filter so same time period as current team.
        if team_end_date:
            df_sto_subset = df_sto_subset[
                df_sto_subset['meta.timestamp'] < team_end_date]

        df_before_opp, df_after_opp = split_dataframe_by_time(
            df_sto_subset, delta_before, delta_after, how='new')

        df_before_opp['team_opp'] = team_opp
        df_after_opp['team_opp'] = team_opp

        baseline_sent_before.append(df_before_opp)
        baseline_sent_after.append(df_after_opp)

    df_all_bef = pd.concat(baseline_sent_before)
    df_all_aft = pd.concat(baseline_sent_after)

    print(df_all_bef['sum_neg'].mean(), df_all_aft['sum_neg'].mean())
    stats.ttest_ind(
        df_all_bef['sum_neg'], df_all_aft['sum_neg'], trim=.2)

    return df_all_bef, df_all_aft


def norm_scores(df_before, df_after, df_base_bef, df_base_aft):
    # difference with mean divided by sigma
    std_bef = np.std(df_base_bef['sum_neg'])
    std_aft = np.std(df_base_aft['sum_neg'])
    mean_bef = np.mean(df_base_bef['sum_neg'])
    mean_aft = np.mean(df_base_aft['sum_neg'])
    df_before['norm_neg'] = (df_before['sum_neg'] - mean_bef)/std_bef
    df_after['norm_neg'] = (df_after['sum_neg'] - mean_aft)/std_aft

    return df_before, df_after


def sent_againt_user_after_criticism(teams, df_sent_twd_user, df_resp_sent,
                                     df_all_comments):
    """
    Return a dataframe that contains the criticism of a user towards other
    users before and after recieveing their first criticism within a certain
    timeframe. Easier with an example: User Alex is criticized by a Boston
    user. This dataframe contains Alex's criticism toward other user in a
    certain timeframe.
    """

    df_neg_from_team = df_sent_twd_user[
        (df_sent_twd_user['neg'] >= .8)].copy()

    # Grab the first criticism towards an author.
    df_neg_from_team.sort_values('meta.timestamp', inplace=True)
    # df_neg_from_team = df_neg_from_team.groupby('author').first().reset_index()
    # This grabs the first criticism from each team.
    df_neg_from_team = df_neg_from_team.groupby(
        ['author', 'neg_from']).first().reset_index()

    # Drop out unneccesary columns
    neg_from_cols = ['author', 'neg_from', 'meta.timestamp']
    df_neg_from_team = df_neg_from_team[neg_from_cols]
    df_neg_from_team.rename(columns={'meta.timestamp': 'cutoff_date'},
                            inplace=True)

    # Add extra columns to df_resp_sent
    df_resp_sent_add = pd.merge(df_all_comments[['name', 'parent_id',
                                                 'meta.timestamp',
                                                 'author', 'team_flair']],
                                df_resp_sent, on='name')

    # Get information about the parent_id
    df_resp_sent_with_parent = pd.merge(df_resp_sent_add,
                                        df_all_comments[[
                                            'name', 'author', 'team_flair',]],
                                        left_on='parent_id', right_on='name')

    df_resp_sent_with_parent.drop(columns=['name_y'], inplace=True)
    df_resp_sent_with_parent.rename(columns={'name_x': 'name',
                                             'author_x': 'author',
                                             'team_flair_x': 'team_flair',
                                             'author_y': 'parent_author',
                                             'team_flair_y': 'parent_team_flair'
                                             }, inplace=True)

    delta_before = datetime.timedelta(weeks=2)
    delta_after = datetime.timedelta(weeks=2)

    df_against = pd.merge(df_resp_sent_with_parent,
                          df_neg_from_team, on='author')

    # Filter to teams that played in the playoffs.
    df_against = df_against[df_against['neg_from'].isin(teams)]

    df_against['time_difference'] = (df_against['cutoff_date'] -
                                     df_against['meta.timestamp']).abs()
    df_before = df_against[(df_against['meta.timestamp']
                            < df_against['cutoff_date']) &
                           (df_against['time_difference']
                            <= delta_before)].copy()
    df_after = df_against[(df_against['meta.timestamp']
                           > (df_against['cutoff_date'])) &
                          (df_against['time_difference']
                           <= delta_after)].copy()

    df_before['time_period'] = 'before'
    df_after['time_period'] = 'after'
    df_before2 = df_before[df_before['parent_team_flair']
                           == df_before['neg_from']]
    df_after2 = df_after[df_after['parent_team_flair']
                         == df_after['neg_from']]
    df_before3 = df_before[df_before['parent_team_flair']
                           != df_before['neg_from']]
    df_after3 = df_after[df_after['parent_team_flair']
                         != df_after['neg_from']]

    df_before2['Criticised by flair'] = 'Flair criticized user'
    df_after2['Criticised by flair'] = 'Flair criticized user'
    df_before3['Criticised by flair'] = 'Flair did Not criticized user'
    df_after3['Criticised by flair'] = 'Flair did Not criticized user'

    df_plt = pd.concat([df_before2, df_after2, df_before3, df_after3])

    return df_plt


def plot_kde_sent_bef_vs_after_criticism(df_plt):
    """
    Plot a kde plot showing the sentiment of users towards a team's fanbase
    before and after being criticized by that fanbase.
    """
    df_plt = df_plt[df_plt['Criticised by flair'] == 'Flair criticized user']

    plt.figure(figsize=(16, 12))
    # Set the color palette with deep red and deep blue
    colors = ["#0000FF", "#FF0000"]
    sns.set_palette(sns.color_palette(colors))
    hue_order = ['before', 'after']

    # Create the KDE plot
    ax = sns.kdeplot(data=df_plt, x='emma_neg', hue='time_period',
                     hue_order=hue_order,
                     fill=True, common_norm=False, legend=True)
    sns.move_legend(ax, loc='upper left', handles=ax.legend_.legendHandles,
                    labels=["Before Criticism",
                            "After Criticism"], title=None, fontsize=26)

    plt.title(
        'Negative Sentiment toward offending fanbase users increases after '
        'criticism',
        fontsize=26)
    # toward offending fanbase users
    plt.xlabel('Toxicity (higher is more toxic)', fontsize=24)
    plt.ylabel('Density', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    # Show the plot
    plt.show()

    return


def plot_bar_sent_bef_vs_after_criticism(df_plt):
    """
    A bar plot comparing the difference towards a team before and after being
    criticized by a flair.
    """
    plt.figure(figsize=(16, 12))
    hue_order = ['before', 'after']
    plt.legend(loc="upper left")
    ax = sns.barplot(data=df_plt, x="Criticised by flair",
                     y="emma_neg",  # "Negativity toward Heat",
                     hue="time_period", hue_order=hue_order, alpha=0.7)

    sns.move_legend(ax, loc='upper left', handles=ax.legend_.legendHandles,
                    labels=["Before Criticism",
                            "After Criticism"], title=None, fontsize=26)

    plt.ylabel("Toxicity", fontsize=24)
    plt.xlabel("", fontsize=16)
    ax.set_xticklabels(["Directed towards aggressor's fanbase",
                        "Directed towards all other fanbases"], fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title(
        "Users become more toxic toward the fanbase of agressors",
        fontsize=26)
    plt.ylim(0.4, 0.70)
    plt.show()


def sent_againt_team_after_criticism(df_sent_twd_user, df_resp_sent,
                                     sentiment_dict, df_all_comments):
    delta_before = datetime.timedelta(weeks=2)
    delta_after = datetime.timedelta(weeks=2)

    teams = [
        'Suns', 'Grizzlies', '76ers', 'Bucks',
        'Mavericks', 'Heat', 'Warriors', 'Celtics'
    ]

    data_plots = list()
    for team in teams:
        df_sent_team = sentiment_dict[team]['df_team']
        sent_team_cols = sentiment_dict[team]['team_cols']

        neg_cols = [c for c in sent_team_cols if 'neg' in c]

        df_neg_from_team = first_critcism_by_team(df_sent_twd_user, team)

        df_sent_team_subset = pd.merge(
            df_sent_team, df_neg_from_team, on='author')

        # Aggregate all negative scores.
        df_sent_team_subset['sum_neg'] = df_sent_team_subset[neg_cols].mean(
            axis=1)

        df_before, df_after = split_dataframe_by_time(
            df_sent_team_subset, delta_before, delta_after, how='prev')

        df_base_bef, df_base_aft = non_treatment_team_sentiment(
            team, teams, sentiment_dict, df_neg_from_team, df_sent_twd_user,
            delta_before, delta_after,
            team_end_date=df_sent_team['meta.timestamp'].max())

        df_before, df_after = norm_scores(
            df_before, df_after, df_base_bef, df_base_aft)

        print(team, stats.ttest_ind(
            df_before['sum_neg'], df_after['sum_neg'], trim=.2))
        print(team, stats.ttest_ind(
            df_before['norm_neg'], df_after['norm_neg'], trim=.2))

        df_scores = pd.DataFrame({
            'score': np.concatenate(
                (df_before['sum_neg'], df_after['sum_neg'])),
            'score_norm': np.concatenate(
                (df_before['norm_neg'], df_after['norm_neg'])),
            'author': np.concatenate(
                (df_before['author'], df_after['author'])),
            'Criticism from Flair': ['Before' for i in df_before['norm_neg']] +
            ['After' for i in df_after['norm_neg']]
        })

        df_scores['Criticized by Flair'] = team

        data_plots.append(df_scores)

    df_plt_all = pd.concat(data_plots)

    return df_plt_all


def plot_sent_againt_team_after_criticism(df_plt_all):
    """
    Plot the negative sentiment toward a team before and after criticism.
    """
    df_order = df_plt_all.groupby(
        'Criticized by Flair')['score'].mean().reset_index()
    df_order.sort_values('score', inplace=True)
    order = df_order['Criticized by Flair'].to_list()

    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 12))

    custom_palette = {"Before": "blue", "After": "red"}

    hue_order = ['Before', 'After']
    ax = sns.barplot(data=df_plt_all, x="Criticized by Flair",
                     y="score",  # "Negativity toward Heat",
                     hue="Criticism from Flair", hue_order=hue_order,
                     palette=custom_palette, alpha=0.7, order=order)
    sns.move_legend(ax, loc='upper left', title=None,
                    handles=ax.legend_.legend_handles,
                    labels=["Before Criticism", "After Criticism"],
                    fontsize=26)

    plt.title("Negative Sentiment Toward Team Before & After Criticism",
              fontsize=26)
    plt.ylim(0.4, 0.6)
    plt.ylabel("Toxicity", fontsize=24)
    plt.xlabel("Offending/Criticizing team", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=22)

    plt.show()


def run_mixed_linear_model_regre(df_plt_all):
    df_plt_all['is_after'] = df_plt_all['Criticism from Flair'].apply(
        lambda x: False if x == 'Before' else True)
    md = smf.mixedlm("score_norm ~ is_after", df_plt_all,
                     groups=df_plt_all["author"])
    mdf = md.fit(method="cg")
    print(mdf.summary())
    return mdf


if __name__ == "__main__":
    teams = [
        'Pelicans', 'Jazz', 'Nuggets', 'Timberwolves',
        'Hawks', 'Raptors', 'Bulls', 'Nets',
        'Suns', 'Grizzlies', '76ers', 'Bucks',
        'Mavericks', 'Heat', 'Warriors', 'Celtics'
    ]

    data_manager = dm.DataManager()

    df_resp_sent = data_manager.load_sentiment('you', 'nba', 'sent_score')

    df_all_comments = process_scores.get_all_comments(data_manager, 4)
    sentiment_dict = process_scores.get_sentiment_scores(
        data_manager, teams, df_all_comments)

    df_sent_twd_user = create_sent_twd_user(df_all_comments, df_resp_sent)

    df_plt = sent_againt_user_after_criticism(teams, df_sent_twd_user,
                                              df_resp_sent, df_all_comments)
    # Plot comparison compared against users.
    plot_kde_sent_bef_vs_after_criticism(df_plt)

    plot_bar_sent_bef_vs_after_criticism(df_plt)

    # Plot comparison compared against team.
    df_plt_all = sent_againt_team_after_criticism(df_sent_twd_user,
                                                  df_resp_sent,
                                                  sentiment_dict, df_all_comments)

    plot_sent_againt_team_after_criticism(df_plt_all)

    # df_plt_all_bef = df_plt_all[df_plt_all['is_before']]
    # df_plt_all_aft = df_plt_all[~df_plt_all['is_before']]

    # df_plt_all_bef = df_plt_all_bef.groupby('author')['score_norm'].mean().reset_index()
    # df_plt_all_aft = df_plt_all_aft.groupby('author')['score_norm'].mean().reset_index()

    # stats.ttest_rel(
    #     df_plt_all_bef['score_norm'], df_plt_all_aft['score_norm'])

    author_bef = set(df_plt_all[
        df_plt_all['Criticism from Flair'] == 'Before']['author'].unique())
    author_aft = set(df_plt_all[
        df_plt_all['Criticism from Flair'] == 'After']['author'].unique())
    df_linear = df_plt_all[df_plt_all['author'].isin(author_bef & author_aft)]

    run_mixed_linear_model_regre(df_linear)
