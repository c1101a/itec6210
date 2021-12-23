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


def rank_docs(q, corpus, top_n):
    docs = corpus

    tokenized_corpus = [doc['text'].split(" ") for doc in docs]
    bm25 = BM25Plus(tokenized_corpus)

    tokenized_query = q.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)

    for i in range(len(docs)):
        docs[i]['score'] = doc_scores[i]

    docs.sort(key=get_score, reverse=True)

    topN = []
    for doc in docs[:top_n]:
        topN.append(doc)
    return topN


txt_list = get_data(PATH_TO_CRAN_TXT, ID_marker)
qry_list = get_data(PATH_TO_CRAN_QRY, ID_marker)
chunk_start = re.compile('\.[A,B,T,W]')
txt_data = []
qry_data = []
rel_data = []

for line in txt_list:
    entries = re.split(chunk_start, line)
    txt_data.append(
        {"id": entries[0].strip(), "title": entries[1], "text": entries[4], "score": 0})

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

final_p = 0
for q in qry_data:
    final_p += get_map(q["query"], q["id"], 10)
map = final_p/len(qry_data)
print(map)
# print(map)
# print(get_map(qry_data[0]["query"], qry_data[0]["id"], 25))
# print(get_map(qry_data[1]["query"], qry_data[1]["id"], 25))
# ## Sentiment
# sia = SentimentIntensityAnalyzer()
# qS = sia.polarity_scores(query)
# qV = [qS['neg'], qS['neu'], qS['pos']]
# for doc in top25:
#     dS = sia.polarity_scores(doc['text'])
#     dV = [dS['neg'], dS['neu'], dS['pos']]
#     #print(1 - spatial.distance.cosine(qV, dV))


