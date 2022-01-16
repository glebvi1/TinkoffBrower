import logging
import multiprocessing as mp
from time import time

import pandas as pd

from data import LIMIT
from model.Document import Document


def build_data():
    start_time = time()
    df_songs = pd.read_csv("data/lyrics_data.csv")
    df_authors = pd.read_csv("data/artists-data.csv")

    documents = multiprocessing_doc(df_songs, df_authors, eda)

    logging.info(f"The count of data = {len(documents)}")
    logging.info(f"Build data in {round(time() - start_time)} seconds.")

    return documents


def eda(df_songs, df_authors):
    df_songs = df_songs[df_songs["Idiom"] == "ENGLISH"].copy()
    df_songs.drop(columns=["SLink", "Idiom"], inplace=True)

    df_authors.drop(columns=["Songs", "Genre", "Genres"], inplace=True)
    df_authors.rename(columns={"Link": "ALink"}, inplace=True)

    df = df_songs.set_index("ALink").join(df_authors.set_index("ALink"))
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["SName", "Lyric", "Artist"], inplace=True)

    documents = []
    count = 0
    for ind in df.index:
        d = Document(df["SName"][ind], df["Lyric"][ind], df["Artist"][ind], df["Popularity"][ind])
        documents.append(d)
        count += 1
        if count == LIMIT:
            break

    return documents


def multiprocessing_doc(df_songs, df_authors, func):
    pool = mp.Pool(4)
    documents = pool.starmap(func, [(df_songs, df_authors)])
    documents = sum(documents, [])
    pool.close()
    pool.join()
    return documents
