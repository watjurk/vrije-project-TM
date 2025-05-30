import os
import pandas as pd


def load():
    DATA_FOLDER = f"{os.getcwd()}/data"
    NER_PATH = f"{DATA_FOLDER}/NER-test.tsv"
    SENTIMENT_TOPIC_PATH = f"{DATA_FOLDER}/sentiment-topic-test.tsv"

    NER = pd.read_csv(NER_PATH, sep="\t")
    SENTIMENT_TOPIC = pd.read_csv(SENTIMENT_TOPIC_PATH, sep="\t")

    return (NER, SENTIMENT_TOPIC)
