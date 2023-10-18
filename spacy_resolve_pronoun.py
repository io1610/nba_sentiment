# -*- coding: utf-8 -*-

from spacy.tokens import Doc
from spacy.language import Language
from fastcoref import spacy_component
import re

from typing import Dict, List
import typing


resolvable_pronouns = {'he', 'her', 'herself', 'him', 'himself', 'she', 'his',
                       'it', 'its', 'itself',  'our', 'ours', 'ourselves',
                       'them', 'themself', 'they', 'us', 'we', 'yours',
                       'yourself', 'yourselves'}


@Language.factory(
    "resolve_fastcoref_pronoun",
    assigns=["doc._.pronoun_resolved_text"],
    default_config={
        "resolvable_pronouns": resolvable_pronouns,
        "resolvable_words": None
    },
)
class FastCorefPronounResolver:
    """Custom class to resolve pronouns in text conversations.
    """

    def __init__(self, nlp, name, resolvable_pronouns: Dict[str, int],
                 resolvable_words: typing.Optional[List[str]] = None):
        self.resolvable_pronouns = resolvable_pronouns
        self.resolvable_words = resolvable_words

        # Register custom extension on the Doc
        if not Doc.has_extension("pron_resolved_text"):
            Doc.set_extension("pron_resolved_text", default="")

    def _find_span_root(self, span):
        pos_noun = ['PROPN', 'NOUN']
        root = span.root
        # Check if root is a Noun
        if root.pos_ in pos_noun:
            return root

        # Else return first noun. If no noun return root.
        for doc in span:
            if doc.pos_ in pos_noun:
                return doc
        return root

    def _get_token_clusters(self, doc):
        coref_clusters = doc._.coref_clusters

        token_coref_cluster, token_head_coref_cluster = list(), list()

        for cluster in coref_clusters:
            token_cluster, token_head_cluster = list(), list()
            for span_idx in cluster:
                if span_idx is None:
                    print('Span is None...')
                    continue
                span = doc.char_span(span_idx[0], span_idx[1])

                token_cluster.append(span)
                token_head_cluster.append(self._find_span_root(span))

            token_coref_cluster.append(token_cluster)
            token_head_coref_cluster.append(token_head_cluster)

        return token_coref_cluster, token_head_coref_cluster

    def _is_resovable_cluster(self, cluster):
        """
        Returns is a cluster is resolvable. A cluster is considered resolvable
        if it contains a resolvable pronoun and a resolvable_word. If
        resolvable_word is None then any non resolvable_pronoun is considered
        valid.
        """
        pos_noun = ['PROPN', 'NOUN']
        if not any([word.text.lower() in self.resolvable_pronouns
                    for word in cluster]):
            return False

        if self.resolvable_words:
            return any([word.text in self.resolvable_words
                        for word in cluster])

        else:
            return any([word.text.lower() not in self.resolvable_pronouns
                        for word in cluster])

    def _find_replacement_word(self, cluster):
        if self.resolvable_words:
            word = next(filter(lambda x: x.text in self.resolvable_words,
                               cluster)).text
        else:
            word = next(filter(lambda x: x.text not in
                               self.resolvable_pronouns, cluster)).text
        return word

    def _set_fastcoref_text(self, doc, print_fix=False):
        tokens = [token for token in doc]

        token_coref_cluster, token_head_coref_cluster = \
            self._get_token_clusters(doc)

        if print_fix:
            print(token_coref_cluster)
            print(token_head_coref_cluster)

        pos_to_noun = dict()
        # Find root coref_
        for cluster in token_head_coref_cluster:
            if not self._is_resovable_cluster(cluster):
                continue

            replace_word = self._find_replacement_word(cluster)

            for token in cluster:
                pronoun_word = token.text.lower()
                if pronoun_word not in self.resolvable_pronouns or \
                        token.text == replace_word or \
                        re.search('[a-zA-Z]', token.text) is None:
                    continue

                pos_to_noun[token.i] = replace_word

        new_text = ''
        for n, token in enumerate(tokens):
            if n in pos_to_noun:
                new_text += pos_to_noun[n] + tokens[n].whitespace_
            else:
                new_text += tokens[n].text_with_ws
        doc._.pron_resolved_text = new_text
        return

    def __call__(self, doc: Doc):
        self._set_fastcoref_text(doc)

        return doc
