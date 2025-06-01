import os
import pandas as pd


def load():
    DATA_FOLDER = f"{os.getcwd()}/data"
    NER_PATH = f"{DATA_FOLDER}/NER-test.tsv"
    SENTIMENT_TOPIC_PATH = f"{DATA_FOLDER}/sentiment-topic-test.tsv"

    NER = pd.read_csv(NER_PATH, sep="\t")
    SENTIMENT_TOPIC = pd.read_csv(SENTIMENT_TOPIC_PATH, sep="\t")

    return (NER, SENTIMENT_TOPIC)


def NER_data_to_sentence_array(NER: pd.DataFrame) -> list[tuple[str, list[tuple[str, str, str]]]]:
    sentences = []

    current_sentence_id = 0
    sentence = ""
    token_ids = []
    tokens = []
    BIO_NER_tags = []

    for i, row in NER.iterrows():
        sentence_id = row["sentence_id"]
        if sentence_id != current_sentence_id:
            sentences.append((sentence, list(zip(tokens, BIO_NER_tags, token_ids))))
            current_sentence_id = sentence_id

            sentence = ""
            tokens = []
            BIO_NER_tags = []
            token_ids = []

        token = row["token"]
        token_id = row["token_id"]
        BIO_NER_tag = row["BIO_NER_tag"]

        if sentence == "":
            sentence = token
        else:
            sentence += f" {token}"
        tokens.append(token)
        BIO_NER_tags.append(BIO_NER_tag)
        token_ids.append(token_id)

    return sentences
