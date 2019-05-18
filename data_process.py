import os
import numpy as np 
import tensorflow as tf
import json

batch_size = 128
seq_length = 70
path_train = "E:\\深度学习\\Dataset\\train.txt"
wiki_vec_path = "E:\\深度学习\\Dataset\\wiki_word2vec_50.bin"


def cat_to_id(classes = None):
    """
    0--pos
    1--neg
    """
    if not classes:
        classes = ['0','1']
    cat2id = {cat:idx for (idx, cat) in enumerate(classes)}
    return cat2id

def build_word2id():
    word_set = ["<PAD>"]
    count = 1
    fi = open(path_train, 'r', encoding= 'UTF-8')
    lines = fi.readlines()
    for line in lines:
        words = line.strip().split()[1:]
        for word in words:
            if word not in word_set:
                word_set.append(word)
                count +=1
    print("vocab_size is %d" % count)
    word2id = dict(zip(word_set, list(range(0, count))))
    id2word = dict(zip(list(range(0,count)), word_set))
    fo1 = open("word2id.txt", "w")
    fo2 = open("id2word.txt", "w")
    fo1.write(json.dumps(word2id))
    fo2.write(json.dumps(id2word))

def build_word2vec(fname, save_to_path):
    import gensim
    f = open("word2id.txt", 'r').read()
    word2id = json.loads(f)
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        np.save(save_to_path, word_vecs)
            

def load_word2vec(fname="data/word2vec.npy"):
    if not os.path.exists(fname):
        build_word2vec(wiki_vec_path, fname)
    return np.load(fname)


def read_corpus(path):
    fi = open(path, 'r', encoding= 'UTF-8')
    f = open("word2id.txt", 'r').read()
    word2id = json.loads(f)
    lines = fi.readlines()
    data = []
    for line in lines:
        label_words = line.strip().split()
        label = int(label_words[0])
        words = label_words[1:]
        sentid = []
        for word in words:
            if word not in word2id.keys():
                word = "<PAD>"
            sentid.append(word2id[word])
        if len(sentid)>seq_length:
            sentid = sentid[:seq_length]
        elif len(sentid)<seq_length:
            sentid += [word2id["<PAD>"]] * (seq_length-len(sentid))
        data.append((sentid,label))  
    return data



if __name__  == '__main__':
    #build_word2vec(wiki_vec_path, "data/word2vec.npy")
    #word2vec = load_word2vec()
    #print(word2vec.shape)
    data = read_corpus("E:\\深度学习\\Dataset\\train.txt")
    print(len(data))
    #build_word2id()
         
    
