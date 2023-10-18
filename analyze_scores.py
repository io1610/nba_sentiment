# -*- coding: utf-8 -*-

import datetime
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

import data_manager as dm
import nba_schedule
import process_scores


def plt_round_three_line_plots(df_all_scores, df_sch, print_info=False):
    delta_after = datetime.timedelta(days=3)

    round_num = 3
    round_three_teams = ['Mavericks', 'Heat', 'Warriors', 'Celtics']
    score_dict = dict()
    for team in round_three_teams:
        score_dict[team] = list()
        df_sch_team = df_sch[(df_sch['team'] == team) &
                             (df_sch['round'] == round_num)].copy()
        df_sch_team.reset_index(drop=True, inplace=True)
        df_sch_team['date_end'] = df_sch_team['date'].shift(-1)
        df_sch_team.loc[df_sch_team.index[-1], 'date_end'] = \
            df_sch_team.iloc[-1]['date'] + delta_after

        df_team_score = df_all_scores[df_all_scores['neg_to'] == team]
        sch_cols = ['date', 'date_end', 'Result']
        for date_st, date_end, result in df_sch_team[sch_cols].values:
            df_tmp = df_team_score[
                (df_team_score['meta.timestamp'] >= date_st) &
                (df_team_score['meta.timestamp'] <= date_end)]
            if print_info:
                print(result, df_tmp.shape, f"{df_tmp['sum_neg'].mean():.3f}")
            score_dict[team].append([result, df_tmp['sum_neg'].mean()])

    team_dicts = [{'Team': team, 'Result': result, 'Value': value}
                  for team, results in score_dict.items()
                  for result, value in results]
    df_up = pd.DataFrame(team_dicts)
    df_up['Game Number'] = df_up.groupby('Team').cumcount() + 1

    # Set Seaborn style
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot 'X' markers
    sns.lineplot(data=df_up, x='Game Number',
                 y='Value', hue='Team', dashes=False, linewidth=3, ax=ax)
    sns.scatterplot(data=df_up, x="Game Number", y="Value",
                    style="Result", markers={"L": "X", "W": "o"},
                    color='k', s=150, zorder=2, ax=ax)

    sns.move_legend(ax, loc='upper left', handles=ax.legend_.legendHandles,
                    title="",
                    fontsize=20)

    # Customize the plot
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title('Negative Team Sentiment per Game (Round 3)', fontsize=26)
    plt.xlabel('Game Number', fontsize=24)
    plt.ylabel('Negative Sentiment', fontsize=24)

    for lh in ax.legend_.legendHandles:
        lh.set_alpha(1)
        lh._sizes = [100]

    plt.show()


def team_most_neg_toward(df_all_scores):
    """
    Return a DataFrame which shows which team is most negative towards another
    team.  Also plot a heat map.
    """
    teams = [
        'Pelicans', 'Jazz', 'Nuggets', 'Timberwolves',
        'Hawks', 'Raptors', 'Bulls', 'Nets',
        'Suns', 'Grizzlies', '76ers', 'Bucks',
        'Mavericks', 'Heat', 'Warriors', 'Celtics'
    ]

    df_tmp = df_all_scores.groupby(['team_flair', 'neg_to'])[
        'sum_neg'].mean().reset_index()

    df_tmp = df_tmp[df_tmp['team_flair'] != 'neg_to']
    df_tmp = df_tmp[df_tmp['team_flair'].isin(teams)]
    df_tmp.reset_index(drop=True, inplace=True)

    df_plt = df_tmp.loc[df_tmp.groupby('team_flair')['sum_neg'].idxmax()]

    pivot_table = pd.pivot_table(df_tmp, values='sum_neg', index='team_flair',
                                 columns='neg_to', fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt=".3f", cbar=True)
    plt.title("Relationship between team users towards Teams (Heatmap)")
    plt.xlabel("Negatively From Team Users")
    plt.ylabel("Team Flair")
    plt.show()

    return df_plt


def get_most_least_liked(sentiment_dict, df_sch):
    def _get_neg_scores(df_cur, neg_cols, team, is_before=True):
        neg_score, neg_count = list(), list()
        for neg_col in neg_cols:
            neg_score.append(df_cur[neg_col].mean())
            neg_count.append((~df_cur[neg_col].isnull()).sum())
        return pd.DataFrame({'ent': [c[:-4] for c in neg_cols],
                             'val': neg_score,
                             'cnt': neg_count,
                             'is_before': [is_before for _ in neg_cols],
                             'team': [team.lower() for _ in neg_cols]
                             })

    data_scores = list()
    for team in sentiment_dict.keys():
        df_sent = sentiment_dict[team]['df_team']
        sent_cols = sentiment_dict[team]['team_cols']

        neg_cols = [c for c in sent_cols if 'neg' in c]
        max_date, min_date = process_scores.first_and_last_date(df_sch, team)

        df_bef = df_sent[df_sent['meta.timestamp'] <= min_date]
        df_aft = df_sent[df_sent['meta.timestamp'] >= max_date]

        df_score_bef = _get_neg_scores(df_bef, neg_cols, team, True)
        df_score_aft = _get_neg_scores(df_aft, neg_cols, team, False)
        data_scores.extend([df_score_bef, df_score_aft])

    df_score = pd.concat(data_scores)
    return df_score


def plot_most_diff(df_score):
    """
    Plot the player with the largest sentiment difference between the start
    and the end of each team's playoff run.
    """

    df = df_score[(df_score['ent'] != df_score['team'])]
    df2 = df.groupby('ent')['cnt'].min().reset_index()
    allowable_ents = df2[df2['cnt'] >= 80]['ent'].unique()

    df = df[df['ent'].isin(allowable_ents)]

    df = df[df['cnt'] >= 80]
    df.reset_index(drop=True, inplace=True)
    df.loc[df.groupby(['team', 'is_before'])['val'].idxmax()]

    df_b = df[df['is_before']]
    df_a = df[~df['is_before']]

    df_diff = pd.merge(df_a, df_b, on=['ent', 'team'])
    df_diff['val_x'] - df_diff['val_y']
    df_diff['diff'] = df_diff['val_x'] - df_diff['val_y']
    df_diff['diff_abs'] = df_diff['diff'].abs()

    df_plt = df_diff.loc[df_diff.groupby('team')['diff_abs'].idxmax()]
    df_plt.sort_values('team', inplace=True, ascending=False)
    df_plt['team_ent'] = df_plt[['team', 'ent']].apply(
        lambda x: f"{x[0].capitalize()} ({x[1].capitalize()})", axis=1)
    df_plt.sort_values('diff_abs', inplace=True, ascending=False)

    plt.figure(figsize=(16, 12))
    plt.hlines(y=df_plt['team_ent'], xmin=df_plt['val_x'],
               xmax=df_plt['val_y'], color='grey', alpha=0.8, linewidth=3)
    plt.scatter(df_plt['val_y'], df_plt['team_ent'], color='blue', alpha=1,
                label='Neg sent before', s=100)
    plt.scatter(df_plt['val_x'], df_plt['team_ent'], color='red', alpha=1,
                label='Neg sent after', s=100)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title("Largest Sentiment Diff Before & After Playoff Run", fontsize=26)
    plt.xlabel("Negative Sentiment", fontsize=24)
    plt.legend(fontsize=24)


def plot_opp_vs_non_opp(teams, sentiment_dict, df_sch):
    """
    Plot a bar plots showing the sentiment toward a team from opp teams that
    the team eliminated vs all other teams.
    """
    opp_data = list()
    for team in teams:
        df_sent = sentiment_dict[team]['df_team']
        opps, eli_dates = process_scores.find_opps_eliminated(df_sch, team)

        opp_idds = list()
        for opp, eli_date in zip(opps, eli_dates):
            df_opp_tmp = df_sent[(df_sent['team_flair'].isin(opps))
                                 & (df_sent['meta.timestamp'] >= eli_date)]
            opp_idds += df_opp_tmp.name.to_list()

        df_opp = df_sent[df_sent['name'].isin(set(opp_idds))]
        df_not_opp = df_sent[(~df_sent['name'].isin(set(opp_idds))) &
                             (df_sent['team_flair'] != team)]
        print(df_opp.shape, df_not_opp.shape)

        neg_col = f'{team.lower()}_neg'

        a = df_opp[~df_opp[neg_col].isnull()][neg_col].values
        b = df_not_opp[~df_not_opp[neg_col].isnull()][neg_col].values
        df_a = pd.DataFrame(a, columns=['score'])
        df_a['is_opp'] = 'Opp'
        df_a['team'] = team

        df_b = pd.DataFrame(b, columns=['score'])
        df_b['is_opp'] = 'Not Opp'
        df_b['team'] = team

        opp_data += [df_a, df_b]

    df_opp_sco_data = pd.concat(opp_data)

    df_order = df_opp_sco_data[df_opp_sco_data['is_opp'] == 'Opp'].groupby(
        'team')['score'].mean().reset_index()
    df_order.sort_values('score', inplace=True)
    order = df_order['team'].to_list()

    sns.set(style="whitegrid")

    plt.figure(figsize=(16, 12))

    custom_palette = {"Not Opp": "blue", "Opp": "red"}
    hue_order = ['Not Opp', 'Opp']

    ax = sns.barplot(data=df_opp_sco_data, x="team", y="score",
                     hue="is_opp", hue_order=hue_order,
                     palette=custom_palette, alpha=1, order=order)
    sns.move_legend(ax, loc='upper left', handles=ax.legend_.legendHandles,
                    labels=["Sentiment from all other fanbases",
                            "Sentiment from fanbase that team eliminated"],
                    title=None, fontsize=22)
    plt.ylim(0.35, 0.55)
    plt.ylabel("Toxicity", fontsize=24)
    plt.xlabel("Team", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title("Sentiment directed at a team from the fanbase it eliminated is "
              "more toxic",
              fontsize=26)

    plt.show()


def plot_cmp_pre_and_post_eli(teams, sentiment_dict, df_sch, print_info=False):
    """
    This creates a dumbbell plot showing the sentiment of teams before and
    after being eliminated.
    """
    flat_sco_data = list()
    for team in teams:
        df_sent = sentiment_dict[team]['df_team']
        sent_cols = sentiment_dict[team]['team_cols']

        max_date, min_date = process_scores.first_and_last_date(df_sch, team)
        max_date = max_date + datetime.timedelta(hours=2, minutes=30)

        df_team_before = df_sent[df_sent['meta.timestamp'] <= min_date]
        df_team_after = df_sent[df_sent['meta.timestamp'] >= max_date]

        neg_cols = [c for c in sent_cols if 'neg' in c]
        neg_before = process_scores.aggregate_scores_all(df_team_before,
                                                         neg_cols)
        neg_after = process_scores.aggregate_scores_all(df_team_after,
                                                        neg_cols)

        pos_cols = [c for c in sent_cols if 'pos' in c]
        pos_before = process_scores.aggregate_scores_all(
            df_team_before, pos_cols)
        pos_after = process_scores.aggregate_scores_all(
            df_team_after, pos_cols)

        if print_info:
            print(team, sentiment_dict[team]['df_team'].shape)
            print(df_team_before.shape, df_team_after.shape)

            print(f"Neg Before: {neg_before:.2f};  After: {neg_after:.2f}")
            print(f"Pos Before: {pos_before:.2f};  After: {pos_after:.2f}")
            print("###########\n")

        neg_arr = df_team_before[neg_cols].to_numpy()
        df_b = pd.DataFrame(neg_arr[~np.isnan(neg_arr)], columns=['score'])
        df_b['time'] = 'before'
        df_b['team'] = team

        neg_arr_a = df_team_after[neg_cols].to_numpy()
        df_a = pd.DataFrame(neg_arr_a[~np.isnan(neg_arr_a)], columns=['score'])
        df_a['time'] = 'after'
        df_a['team'] = team

        flat_sco_data += [df_b, df_a]

    df_flat_sco_data = pd.concat(flat_sco_data)

    plot_type = 'dumbdell'
    if plot_type == 'line':
        sns.set(style="whitegrid")

        plt.figure(figsize=(16, 12))
        plt.title("Negative Team sentiment before and after elimination")

        sns.barplot(data=df_flat_sco_data, x="team", y="score",
                    hue="time")

        plt.xticks(rotation=90)
        plt.legend(loc="upper left")

        plt.show()
    else:
        df_plt = df_flat_sco_data.groupby(['time', 'team'])[
            'score'].mean().reset_index()

        df_a = df_plt[df_plt['time'] == 'before']
        df_b = df_plt[df_plt['time'] == 'after']
        df_plt = pd.merge(df_a, df_b, on='team')
        df_plt['abs_diff'] = abs(df_plt['score_y'] - df_plt['score_x'])
        df_plt.sort_values('abs_diff', inplace=True, ascending=False)

        plt.figure(figsize=(16, 12))
        fig, ax = plt.subplots(figsize=(16, 12))

        plt.hlines(y=df_plt['team'], xmin=df_plt['score_x'],
                   xmax=df_plt['score_y'], color='grey',
                   alpha=0.8, linewidth=3)
        plt.scatter(df_plt['score_x'], df_plt['team'], color='blue',
                    s=100, label='Neg sent before')
        plt.scatter(df_plt['score_y'], df_plt['team'], color='red', s=100,
                    label='Neg sent after')

        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.title("Sentiment Diff Before & After Playoff Run", fontsize=26)
        plt.xlabel("Negative Sentiment", fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        plt.show()


def find_highest_neg(teams, sentiment_dict):
    """
    Returns a DataFrame with each teams most negative sentiment player.
    """
    for team in teams:
        df_sent = sentiment_dict[team]['df_team']
        sent_cols = sentiment_dict[team]['team_cols']

        neg_cols = [c for c in sent_cols if 'neg' in c]
        neg_vals = [np.mean(df_sent[c]) for c in neg_cols]
        idx = np.argmax(neg_vals)
        print(f"team: {team:12} col: {neg_cols[idx]:12}"
              f"  val: {neg_vals[idx]:.2f}")


def get_most_negative_sent_player(teams, sentiment_dict):
    df_sent = sentiment_dict['Celtics']['df_team']
    df_play_neg = df_sent[df_sent['team_flair'].isin(
        teams)]['team_flair'].copy()
    df_play_neg = df_play_neg.drop_duplicates().reset_index(drop=True)

    for team in teams:

        df_sent = sentiment_dict[team]['df_team']
        sent_cols = sentiment_dict[team]['team_cols']
        neg_cols = [
            c for c in sent_cols if 'neg' in c and c[:-4] != team.lower()]

        df_sent2 = df_sent[df_sent['team_flair'].isin(teams)]
        df_play_sent = df_sent2.groupby(
            'team_flair')[neg_cols].mean().reset_index()
        df_play_sent2 = df_sent2.groupby(
            'team_flair')[neg_cols].count().reset_index()

        rename_dict = {c: c[:-4]+f"_{team.lower()}" for c in neg_cols}
        rename_dict2 = {c: c[:-4]+f"_{team.lower()}_cnt" for c in neg_cols}
        df_play_sent.rename(columns=rename_dict, inplace=True)
        df_play_sent2.rename(columns=rename_dict2, inplace=True)
        df_play_sent = pd.merge(df_play_sent, df_play_sent2, on='team_flair')

        df_play_neg = pd.merge(df_play_neg, df_play_sent,
                               on='team_flair', how='left')

    neg_sent_player = list()
    cnt_cols = [c for c in df_play_neg.columns if c[-4:] == '_cnt']
    for team in teams:
        cnt_cols = [c for c in df_play_neg.columns if c[-4:] == '_cnt' and
                    c.split('_')[-2] != team.lower()]
        use_cols = list()
        for c in cnt_cols:
            val = df_play_neg.loc[df_play_neg['team_flair'] == team, c].iloc[0]
            if val >= 100:
                use_cols.append(c[:-4])
        player = df_play_neg[df_play_neg['team_flair']
                             == team][use_cols].idxmax(axis=1).iloc[0]
        neg_sent_player.append([team, player])

    df_neg_sent_player = pd.DataFrame(
        neg_sent_player, columns=['team', 'neg_player'])
    df_neg_sent_player.sort_values('team', inplace=True)
    return df_neg_sent_player


if __name__ == '__main__':
    teams = [
        'Pelicans', 'Jazz', 'Nuggets', 'Timberwolves',
        'Hawks', 'Raptors', 'Bulls', 'Nets',
        'Suns', 'Grizzlies', '76ers', 'Bucks',
        'Mavericks', 'Heat', 'Warriors', 'Celtics'
    ]

    data_manager = dm.DataManager()
    # flair_to_team = data_manager.load_pickle_flair_to_team()
    df_all_comments = process_scores.get_all_comments(data_manager, 4)

    sentiment_dict = process_scores.get_sentiment_scores(data_manager, teams,
                                                         df_all_comments)
    df_sch = nba_schedule.getSchedule(2022)

    # Plot sentiment before and after playoff run.
    plot_cmp_pre_and_post_eli(teams, sentiment_dict, df_sch)

    # Plot opp vs non opp.
    teams_opp_vs_non_opp = ['Suns', 'Grizzlies', '76ers', 'Bucks',
                            'Mavericks', 'Heat',  'Celtics', 'Warriors']
    plot_opp_vs_non_opp(teams_opp_vs_non_opp, sentiment_dict, df_sch)

    # Get most negative players per team.
    get_most_negative_sent_player(teams)

    # Get the least liked player.
    df_scores = get_most_least_liked(sentiment_dict, df_sch)
    plot_most_diff(df_scores)

    df_all_scores = process_scores.all_scores_get_dataframe(sentiment_dict)
    plt_round_three_line_plots(df_all_scores, df_sch)
