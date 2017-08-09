#%%
import spacy 
import json
import numpy as np
import gensim

nlp = spacy.load('en')

class MySentences(object):
    def __init__(self,fname,capital=True):
        self.fname = fname
        self.capital = capital
    def __iter__(self):
        with open(self.fname,'r') as text_file :
            if self.capital == True:
                for line in text_file:
                    yield line.split()
            else:
                for line in text_file:
                    yield line.lower().split()

#%%
def write_to_file(multi_sentences,f):
    if len(multi_sentences)>0:
        for sentence in multi_sentences:
            for word in sentence:
                line = word_text = word.text + ' ' + word.tag_ +  ' ' + word.pos_  + ' ' + '\n'
                f.write(line)
        f.write('\n')


"""
1.change normal text to conll format text
2.control the sentence length(30,max_length=30)
3.delete the influence of "\\"
"""
def text_conll_format(input_text_file_name,out_put_file_name,max_length=30):
    input_text_file = open(input_text_file_name,'r')
    main_text = input_text_file.read()
    input_text_file.close()
    main_text = main_text.replace("\\"," ")

    out_put_file    = open(out_put_file_name,'w+')
    doc = nlp(main_text)
    sents_list = [sent for sent in doc.sents]
    sents = doc.sents
    multi_sentences= []
    num_words = 0
    total = 0

    for sent in sents_list:
        if len(sent) > 94:
            write_to_file(multi_sentences,out_put_file)
            multi_sentences= []
            total = total + num_words
            num_words = 0

            puncs_idx = [] 
            for idx,word in enumerate(sent):
                if word.text in [',','?','.','!'] and idx<=94:
                    puncs_idx.append(idx)
            max_punc_idx = max(puncs_idx)
            multi_sentences.append(sent[0:max_punc_idx])
            write_to_file(multi_sentences,out_put_file)
            multi_sentences = []
            multi_sentences.append(sent[max_punc_idx:])
            write_to_file(multi_sentences,out_put_file)
            multi_sentences= []
            total = total + len(sent)

        elif len(sent) > max_length and len(sent) <= 94:
            write_to_file(multi_sentences,out_put_file)
            multi_sentences= []
            total = total + num_words
            num_words = 0
            
            multi_sentences.append(sent)
            write_to_file(multi_sentences,out_put_file)
            multi_sentences= []
            total = total + len(sent)


        elif num_words + len(sent) > max_length:
            write_to_file(multi_sentences,out_put_file)
            multi_sentences= []
            total = total + num_words

            num_words = len(sent)
            multi_sentences= []
            multi_sentences.append(sent)

            if sent.text == sents_list[-1].text:
                print('zzzzzzzzzzzzzzzzzzzzzzzzzz')
                write_to_file(multi_sentences,out_put_file)
                total = total + num_words

        else:
            multi_sentences.append(sent) 
            num_words = num_words + len(sent)
            if sent.text == sents_list[-1].text:
                print('ggggggggggggggggggggggg')
                write_to_file(multi_sentences,out_put_file)
                total = total + num_words
    print(total)

def generate_gensim_train_sentence(conll_file_name,output_file_name):
    f = open(conll_file_name,'r')
    lines = f.readlines()
    f.close()
    output_file = open(output_file_name,'w+')   
    sentence = [] 
    for line in lines:
        try:
            assert (len(line.split()) == 3)
            word = line.split()[0]
            sentence.append(word)
        except:
            pass
    output_file.write(' '.join(sentence)+'\n')
    output_file.close()

def train_gensim_word2vec_with_new_sentence(gensim_model_name):
    new_sentence = MySentences('data/temp/gensim_train.txt',capital=True)
    model = gensim.models.Word2Vec.load(gensim_model_name)
    model.build_vocab(new_sentence,update=True)
    model.intersect_word2vec_format('data/word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)    
    model.train(new_sentence,total_examples = 1,epochs=10)
    model.save(gensim_model_name)

def label_2(tag):
    one_hot = np.zeros(2)
    if tag.endswith('O'):
        one_hot[0] =1
    else:
        one_hot[1] =1
    return one_hot

def capital_in_word(word):
    label = False 
    for letter in word[1:]:
        if ord('A') <= ord(letter) <= ord('Z'):
            label = True

    if label == True:
        return np.array([1])
    else:
        return np.array([0])

def pos(tag):
    one_hot = np.zeros(6)
    if 'NNP' in tag :
        one_hot[0] = 1
    elif 'VB' in tag:
        one_hot[1] = 1
    elif tag == 'NN' or tag == 'NNS' :
        one_hot[2] = 1
    elif tag == 'CD':
        one_hot[3] = 1
    elif tag == 'JJ':
        one_hot[4] = 1
    else:
        one_hot[5] = 1
    return one_hot

def chunk(tag):
    one_hot = np.zeros(5)
    if 'ADP' in tag:
        one_hot[0] = 1
    elif 'PUNCT' in tag:
        one_hot[1] = 1
    elif 'ADV' in tag:
        one_hot[2] = 1
    elif tag == 'PRON':
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot

def get_word2vec_embeddings(gensim_model_name,conll_file_name,capital = True):
    model = gensim.models.Word2Vec.load(gensim_model_name)
    f = open(conll_file_name,'r')
    lines = f.readlines()
    f.close()
    text_data = []
    train_data = []

    text_sentence = []
    sentence = []

    for line in lines:   
        if 'SP SPACE' in line :
            continue      
        if line in ['\n', '\r\n'] :
            if len(sentence) != 0:
                text_data.append(text_sentence)
                train_data.append(np.array(sentence))
                sentence = []
                text_sentence = []
            else:
                continue
            
        else:
            try:
                assert (len(line.split()) == 3)
            except:
                print(line)
                continue
            line = line.split()
            word = line[0]
            pos_tag = line[1]
            chunk_tag = line[2]

            try:
                if capital == True:
                    word_embedding = model.wv[word]
                else:
                    word_embedding = model.wv[word.lower()]
                pos_embedding = pos(pos_tag)
                chunk_embedding = chunk(chunk_tag)
                capital_embedding = capital_in_word(word)
                embedding = np.append(word_embedding,pos_embedding)
                embedding = np.append(embedding,chunk_embedding)
                embedding = np.append(embedding,capital_embedding)
                text_sentence.append(word)
                sentence.append(embedding)
            except:
                print(line)
    return train_data,text_data




#%%
#text_conll_format('data/temp/temp.txt','data/temp/temp_conll.txt')
#generate_gensim_train_sentence('data/temp/temp_conll.txt','data/temp/gensim_train.txt')

