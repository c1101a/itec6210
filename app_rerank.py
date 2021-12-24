from numpy.linalg import norm
from numpy import dot
import numpy as np
from scipy import spatial
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from rank_bm25 import BM25Plus
import nltk
nltk.download(["vader_lexicon"])

PATH_TO_CRAN_TXT = './data/cran.all.1400'
PATH_TO_CRAN_QRY = './data/cran.qry'
PATH_TO_CRAN_REL = './data/cranqrel'

ID_marker = re.compile('\.I.')


def get_data(PATH_TO_FILE, marker):
    with open(PATH_TO_FILE, 'r') as f:
        text = f.read().replace('\n', " ")
        lines = re.split(marker, text)
        lines.pop(0)
    return lines


def is_relevant(q_id, d_id, rel_data):
    queries = filter(lambda c: c["q_id"] == str(q_id), rel_data)
    for q in list(queries):
        if q["d_id"] == d_id:
            return 1
    return 0


def get_score(doc):
    return doc.get('score')


def get_sim(doc):
    return doc.get('sim')

def get_final(doc):
    return doc.get('final')

def rank_docs(q, corpus, top_n):
    docs = corpus
    sia = SentimentIntensityAnalyzer()
    qS = sia.polarity_scores(q)

    tokenized_corpus = [doc['text'].split(" ") for doc in docs]
    bm25 = BM25Plus(tokenized_corpus)

    tokenized_query = q.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)

    for i in range(len(docs)):
        docs[i]['score'] = doc_scores[i]

    docs.sort(key=get_score, reverse=True)

    topN_docs = docs[:top_n]
    for doc in topN_docs:
        dS = sia.polarity_scores(doc['text'])
        doc["sim"] = abs(qS["compound"] - dS["compound"])

    doc_ranks = []
    for i in range(len(topN_docs)):
        doc_ranks.append(
            {"id": topN_docs[i]["id"], "bm25_rank": i, "sent_rank": 0, "final": 0})
    
    topN_docs.sort(key=get_sim, reverse=False)
    for i in range(len(topN_docs)):
        for doc in doc_ranks:
            if topN_docs[i]["id"] == doc["id"]:
                doc["sent_rank"] = i


    for doc in doc_ranks:
        doc["final"] = (3*doc["bm25_rank"] + .2*doc["sent_rank"])
    
    doc_ranks.sort(key=get_final, reverse=False)

    return doc_ranks


def get_map(query, query_id, top_n):
    index = 0
    positive = 0
    total_precision = 0
    for i in rank_docs(query, txt_data, top_n):
        index += 1
        if is_relevant(query_id, i["id"], rel_data) == 1:
            positive += 1
            total_precision += (positive/index)
    return total_precision/index


txt_list = get_data(PATH_TO_CRAN_TXT, ID_marker)
qry_list = get_data(PATH_TO_CRAN_QRY, ID_marker)
chunk_start = re.compile('\.[A,B,T,W]')
txt_data = []
qry_data = []
rel_data = []

for line in txt_list:
    entries = re.split(chunk_start, line)
    txt_data.append(
        {"id": entries[0].strip(), "title": entries[1], "text": entries[4], "score": 0, "sim": 0})

q_index = 0
for line in qry_list:
    q_index += 1
    entries = re.split(chunk_start, line)
    qry_data.append(
        {"id": q_index, "query": entries[1]})

with open(PATH_TO_CRAN_REL, 'r') as f:
    for line in f:
        l = line.split(" ")
        rel_data.append({"q_id": l[0], "d_id": l[1], "rel": l[2]})

final_p = 0
for q in qry_data:
    final_p += get_map(q["query"], q["id"], 100)
map = final_p/len(qry_data)
print(map)
