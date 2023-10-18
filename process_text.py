# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import markdown
import re
from tqdm import tqdm

tqdm.pandas()


def md_to_text(md):
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


def remove_urls(text):
    # Replace floating urls
    return re.sub('http://\S+|https://\S+', '', text)


def clean_text(text):
    text = md_to_text(text)
    text = remove_urls(text)
    return text


def has_entity_df(df, word_list, is_ignorecase=True, text_col='body'):
    """
    Return bool checking if any word in the list exists in the text
    """
    pattern = r'\b(?:' + '|'.join(map(re.escape, word_list)) + r')\b'
    if is_ignorecase:
        words_re = re.compile(pattern, re.IGNORECASE)
    else:
        words_re = re.compile(pattern)

    return df[text_col].progress_apply(lambda x: not
                                       words_re.search(x) is None)


# Split df_comments into groups around the size of max_num_comments. All
# comments in a groups must be around the same size.
def get_doc_ids(df_comments, max_num_comments=10000):
    group_link_ids = list()

    link_to_cnt = df_comments['link_id'].value_counts().to_dict()

    cur_link_ids = list()
    comment_cnt = 0
    print("MAX_NUM_COMMENTS: ", max_num_comments)
    for link_id in df_comments['link_id'].unique():
        if comment_cnt + link_to_cnt[link_id] > max_num_comments \
                and cur_link_ids:
            group_link_ids.append(cur_link_ids)
            cur_link_ids = []
            comment_cnt = 0

        cur_link_ids.append(link_id)
        comment_cnt += link_to_cnt[link_id]

    group_link_ids.append(cur_link_ids)

    return group_link_ids
