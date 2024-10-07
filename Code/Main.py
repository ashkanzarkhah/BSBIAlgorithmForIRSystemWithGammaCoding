# First we load our documents one by one and build our dictionary
# To do so, we need to tokenize each document first and do all preprocessings on it
# and then append new tokens to our dictionary
# So first we define a function to tokenize a document

import math
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()


def tokenizer(document):
    # First we lower all our words
    document = document.lower()

    # Now we replace shortened forms back to their original form
    shortened_words_dic = {
        'won’t': ' will not',
        'can’t': ' can not',
        '’ve': ' have',
        '’m': ' am',
        '’d': ' would',
        '’ll': ' will',
        '’s': ' is',
        '’re': ' are',
        'n’t': ' not'
    }
    for key, value in shortened_words_dic.items():
        document = document.replace(key, value)

    # Now we tokenize it with nltk word tokenizer
    tokenized_document = word_tokenize(document)

    # Now we remove all punctuation marks from our tokens
    bad_tokens = [',', '?', '”', ')', '’', ';', '“', '.', '$', '(', ':', '!']
    tokenized_document = [
        token for token in tokenized_document if token not in bad_tokens]

    # Now we remove punctuation marks that are left in the middle or at the end of tokens
    tokenized_document = [token.rstrip('.') for token in tokenized_document]
    tokenized_document = [token.rstrip('?') for token in tokenized_document]
    tokenized_document = [token.rstrip('—') for token in tokenized_document]
    tokenized_document = [token.rstrip('-') for token in tokenized_document]
    tokenized_document = [token.rstrip('–') for token in tokenized_document]

    new_tokenized_document = []
    for token in tokenized_document:
        if '.' in token:
            if (token.split('.')[0][:-1] > '9' or token.split('.')[-1][0] > '9'):
                new_tokenized_document.extend(token.split('.'))
            else:
                new_tokenized_document.append(token)
        elif '?' in token:
            new_tokenized_document.extend(token.split('?'))
        elif '-' in token:
            new_tokenized_document.extend(token.split('-'))
        elif '—' in token:
            new_tokenized_document.extend(token.split('—'))
        elif '–' in token:
            new_tokenized_document.extend(token.split('–'))
        else:
            new_tokenized_document.append(token)
    tokenized_document = new_tokenized_document

    # Now we remove stop words from it
    stop_words = ['he', 'for', 'in', 'is',
                  'was', 'of', 'and', 'to', 'a', 'the']
    tokenized_document = [
        token for token in tokenized_document if token not in stop_words]

    # Now its time to convert our tokens to their stemmed forms
    stemmed_tokenized_document = [porter_stemmer.stem(
        token) for token in tokenized_document]

    # Now that we are done with our tokenizing and preprocessing the tokens, it's time to return them
    return stemmed_tokenized_document

# Now it's time to lead documents one by one from disk and build our dictionary


folder_path = 'docs'
files = os.listdir(folder_path)

all_tokens = set()

for file in files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r', encoding="Windows-1252") as file:
        document = file.read()

        new_tokens = tokenizer(document)
        for token in new_tokens:
            all_tokens.add(token)
print(all_tokens)

# Now before implementing BSBI algorithm we first need
# a hash function to get termID from each term
all_tokens_hash = len(all_tokens) * ['#']


def hash_func(token):
    B = 256
    M = len(all_tokens)
    cur = 0
    for i in range(len(token)):
        cur = ((cur * B) + ord(token[i])) % M
    while (all_tokens_hash[cur] != token and all_tokens_hash[cur] != '#'):
        cur = (cur + 1) % M
    if (all_tokens_hash[cur] == '#'):
        all_tokens_hash[cur] = token
    return cur


# Now because for building our inverted indexes
# we are using gamma coding, first we need a gamma encoder, a gamma decoder
# and we need a function to insert another number's code at the end of our code(merger)

def gamma_encoder(inp):
    inp += 1
    len_inp = math.floor(math.log2(inp))

    length = (1 << (len_inp + 1)) - 2
    offset = inp - (1 << len_inp)

    code = (length << len_inp) + offset
    return code


def gamma_decoder(inp):
    answer = []
    cur = math.floor(math.log2(inp))

    while (cur >= 0):
        cnt = 0
        while (inp & (1 << (cur - cnt)) != 0):
            cnt += 1

        cur -= cnt
        cur_num = (1 << cnt) + ((inp % (1 << cur)) >> (cur - cnt))

        cur -= cnt + 1
        answer.append(cur_num - 1)

    return answer


def merger(cur, new):
    if (new == 0):
        cur <<= 1
    else:
        cur <<= math.floor(math.log2(new)) + 1
    cur += new
    return cur

# Now to simulate our disk space, we define a disk class and
# save and load blocks from it


class DISK:
    def __init__(self):
        self.inverted_index_blocks = []
        self.document_frequency_blocks = []
        self.last_document_blocks = []

    def ADD(self, inverted_index, document_frequency, last_document):
        self.inverted_index_blocks.append(inverted_index)
        self.document_frequency_blocks.append(document_frequency)
        self.last_document_blocks.append(last_document)

    def LOAD(self, id):
        return (self.inverted_index_blocks[id],
                self.document_frequency_blocks[id],
                self.last_document_blocks[id])

# Now that we have prepared everything, it's time to read all documents
# again and build gamma coded inverted index with BSBI algorithm
# we set each block to be 5 documents


disk = DISK()
doc_id = 0
inverted_index = [0 for i in range(len(all_tokens))]
document_frequency = [0 for i in range(len(all_tokens))]
last_document = [0 for i in range(len(all_tokens))]

folder_path = 'docs'
files = os.listdir(folder_path)

for file in files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r', encoding="Windows-1252") as file:
        document = file.read()

        doc_id += 1
        new_tokens = tokenizer(document)
        new_tokens.sort()

        ls_cnt = 0
        for i in range(len(new_tokens)):
            token = new_tokens[i]
            ls_cnt += 1

            if (i + 1 == len(new_tokens)) or (new_tokens[i] != new_tokens[i + 1]):
                term_id = hash_func(token)

                document_frequency[term_id] = merger(document_frequency[term_id],
                                                     gamma_encoder(ls_cnt))
                inverted_index[term_id] = merger(inverted_index[term_id],
                                                 gamma_encoder(doc_id - last_document[term_id]))
                last_document[term_id] = doc_id

                ls_cnt = 0

        if doc_id % 5 == 0:
            disk.ADD(inverted_index, document_frequency, last_document)

            inverted_index = [0 for i in range(len(all_tokens))]
            document_frequency = [0 for i in range(len(all_tokens))]
            last_document = [0 for i in range(len(all_tokens))]

# Now that we are sure of our blocks, its time to merge them


def first_num_decreaser(inp, val):
    cur = math.floor(math.log2(inp))
    cnt = 0
    while (inp & (1 << (cur - cnt)) != 0):
        cnt += 1
    cur -= cnt
    cur_num = (1 << cnt) + ((inp % (1 << cur)) >> (cur - cnt))

    cur -= cnt
    ans = (gamma_encoder(cur_num - 1 - val) << cur) + inp % (1 << cur)
    return ans


base_inverted_index = [0 for i in range(len(all_tokens))]
base_document_frequency = [0 for i in range(len(all_tokens))]
base_last_document = [0 for i in range(len(all_tokens))]

for i in range(3):
    inverted_index, document_frequency, last_document = disk.LOAD(i)

    for j in range(len(all_tokens)):
        token = all_tokens_hash[j]
        if (inverted_index[j] != 0):
            new_inverted_index = first_num_decreaser(
                inverted_index[j], base_last_document[j])
            base_inverted_index[j] = merger(base_inverted_index[j],
                                            new_inverted_index)

            base_document_frequency[j] = merger(base_document_frequency[j],
                                                document_frequency[j])

            base_last_document[j] = last_document[j]

# Now at last we just check if our all inverted indexes are correct too
for i in range(len(all_tokens)):
    if (base_inverted_index[i] != 0):
        print(all_tokens_hash[i])
        print(gamma_decoder(base_inverted_index[i]))
        print(gamma_decoder(base_document_frequency[i]))
        print(base_last_document[i])
