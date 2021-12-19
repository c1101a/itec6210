import re
from collections import defaultdict

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
txt_data = defaultdict(dict)

for line in txt_list:
    entries = re.split(chunk_start, line)
    id = entries[0].strip()
    title = entries[1]
    author = entries[2]
    publication_date = entries[3]
    text = entries[4]
    txt_data[id]['title'] = title
    txt_data[id]['author'] = author
    txt_data[id]['publication_date'] = publication_date
    txt_data[id]['text'] = text
