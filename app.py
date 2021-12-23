from scipy import spatial
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from rank_bm25 import BM25Okapi
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


def rank_docs(q, txt_data, top_n):
    tokenized_query = q.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)

    for i in range(len(txt_data)):
        txt_data[i]['score'] = doc_scores[i]

    txt_data.sort(key=get_score, reverse=True)

    topN = []
    for doc in txt_data[:top_n]:
        topN.append(doc)

    return topN


txt_list = get_data(PATH_TO_CRAN_TXT, ID_marker)
qry_list = get_data(PATH_TO_CRAN_QRY, ID_marker)

chunk_start = re.compile('\.[A,B,T,W]')
txt_data = []
rel_data = []

for line in txt_list:
    entries = re.split(chunk_start, line)
    id = entries[0].strip()
    title = entries[1]
    author = entries[2]
    publication_date = entries[3]
    text = entries[4]

    txt_data.append({"id": id, "title": title, "text": text, "score": 0})

with open(PATH_TO_CRAN_REL, 'r') as f:
    for line in f:
        l = line.split(" ")
        rel_data.append({"q_id": l[0], "d_id": l[1], "rel": l[2]})

tokenized_corpus = [doc['text'].split(" ") for doc in txt_data]
bm25 = BM25Okapi(tokenized_corpus)

query = "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft"

map = 0
index = 0
positive = 0
total_precision = 0
for i in rank_docs(query, txt_data, 110):
    index += 1
    if is_relevant(1, i["id"], rel_data) == 1:
        positive += 1
        total_precision += (positive/index)
print(total_precision/positive)

# ## Sentiment
# sia = SentimentIntensityAnalyzer()
# qS = sia.polarity_scores(query)
# qV = [qS['neg'], qS['neu'], qS['pos']]
# for doc in top25:
#     dS = sia.polarity_scores(doc['text'])
#     dV = [dS['neg'], dS['neu'], dS['pos']]
#     #print(1 - spatial.distance.cosine(qV, dV))
