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
    """
    Reads the file and splits the text into entries at the ID marker '.I'.
    The first entry is empty, so it is removed.
    'marker' contains the regex at which we want to split
    """
    with open(PATH_TO_FILE, 'r') as f:
        text = f.read().replace('\n', " ")
        lines = re.split(marker, text)
        lines.pop(0)
    return lines


txt_list = get_data(PATH_TO_CRAN_TXT, ID_marker)
qry_list = get_data(PATH_TO_CRAN_QRY, ID_marker)

chunk_start = re.compile('\.[A,B,T,W]')
txt_data = []

for line in txt_list:
    entries = re.split(chunk_start, line)
    id = entries[0].strip()
    title = entries[1]
    author = entries[2]
    publication_date = entries[3]
    text = entries[4]

    txt_data.append({"id": id, "title": title, "text": text, "score": 0})
    # txt_data[id]['title'] = title
    # txt_data[id]['author'] = author
    # txt_data[id]['publication_date'] = publication_date
    # txt_data[id]['text'] = text
    # txt_data[id]['score'] = 0


tokenized_corpus = [doc['text'].split(" ") for doc in txt_data]
bm25 = BM25Okapi(tokenized_corpus)

query = "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)

for i in range(len(txt_data)):
    txt_data[i]['score'] = doc_scores[i]


def get_score(doc):
    return doc.get('score')


txt_data.sort(key=get_score, reverse=True)

top25 = []
for doc in txt_data[:25]:
    top25.append(doc)

sia = SentimentIntensityAnalyzer()
qS = sia.polarity_scores(query)
qV = [qS['neg'], qS['neu'], qS['pos']]
for doc in top25:
    dS = sia.polarity_scores(doc['text'])
    dV = [dS['neg'], dS['neu'], dS['pos']]
    print(1 - spatial.distance.cosine(qV, dV))



# import and parse the evaluation. combine everything and export to excel
