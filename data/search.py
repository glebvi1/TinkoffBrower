from __future__ import annotations

import logging
from time import time
from typing import Union, List, Tuple

from gensim.utils import tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from data import COUNT_CANDIDATES, COUNT_ALL
from data.data_handler import build_data
from model.Document import Document

logging.basicConfig(level=logging.INFO)
index = []
inverted_index = {}
stemmer = SnowballStemmer(language="english")
vectorizer = TfidfVectorizer(min_df=1)


def build_index() -> None:
    """Создаем лист докуменов; строим инвертированный индекс и сортируем его по популярности автора песни"""
    global index
    start_time = time()
    index = []
    index = build_data()
    for did, doc in enumerate(index):
        text = stemmer_str(doc.title)
        text.extend(stemmer_str(doc.author))

        for word in text:
            if word not in inverted_index:
                inverted_index[word] = []
            inverted_index[word].append((doc, did))

    logging.info(f"Build inverted index in {round(time() - start_time)} seconds.")
    start_time = time()
    for word in inverted_index.keys():
        inverted_index[word].sort(key=lambda tup: -tup[0].popularity)
    logging.info(f"Sort inverted index in {round(time() - start_time)} seconds.")


def score(query, document) -> float:
    """Ищем косинусное сходство между запросом, заголовком и автором документа после преобразование TF-IDF
    :param query: запрос, к которому применили стемминг
    :param document: документ
    """
    stem_query = query
    stem_title = stemmer_str(document.title, return_list=False)
    stem_author = stemmer_str(document.author, return_list=False)

    tfidf_title = vectorizer.fit_transform([stem_query, stem_title])
    tfidf_author = vectorizer.fit_transform([stem_query, stem_author])

    csim = max(cosine_sim(tfidf_author), cosine_sim(tfidf_title))

    return round(csim, 2)


def cosine_sim(tfidf) -> float:
    """Вычисляем косинусное сходство"""
    return (tfidf * tfidf.T).toarray()[0, 1]


def stemmer_str(text: str, return_list=True) -> Union[list, bool]:
    """Применяем стемминг для строки
    :param text: строка, для которой делаем стемминг
    :param return_list: если True, возращаем список из слов; иначе - строку
    """
    new_text = list(tokenize(text, lowercase=True, deacc=True))
    stemmer_word = [stemmer.stem(word) for word in new_text]
    return stemmer_word if return_list else " ".join(stemmer_word)


def retrieve(query) -> List[Tuple[Document, float]]:
    """Подбираем самые релевантные документы к запросу
    :param query: запрос
    """
    if query == "":
        return []

    start_time = time()

    candidates = []
    scored = []
    all_candidates = []
    count_all = 0
    count_candidates = 0

    stem_query = stemmer_str(query)

    for word in stem_query:
        flag = False
        if word in inverted_index:
            for doc, did in inverted_index[word]:
                if len(list(filter(lambda x: x[1] == did, all_candidates))) != 0:
                    continue
                all_candidates.append((doc, did))
                count_all += 1
                if count_all == COUNT_ALL:
                    logging.info(f"All count of candidates more than {COUNT_ALL}")
                    flag = True
                    break
        if flag:
            break

    logging.info(f"Build all candidates in {round(time() - start_time)} seconds.")
    start_time = time()

    stem_query = " ".join(stem_query)

    for doc, did in all_candidates:
        candidates.append(index[did])
        scored.append(score(stem_query, doc))

    logging.info(f"Build score in {round(time() - start_time)} seconds.")
    start_time = time()

    x = dict(zip(candidates, scored))
    x = {k: v for k, v in sorted(x.items(), key=lambda item: -item[1])}

    logging.info(f"Sort score in {round(time() - start_time)} seconds.")
    start_time = time()

    result = []
    for k, v in x.items():
        count_candidates += 1
        result.append((k, v))
        if count_candidates == COUNT_CANDIDATES:
            logging.info(f"The count of relevated candidates more than {COUNT_CANDIDATES}")
            break
    logging.info(f"Build result in {round(time() - start_time)} seconds.")
    return result
