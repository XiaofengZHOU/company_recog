#%%
import os 
import pickle
import numpy as np 
from pprint import pprint
import gensim 

def test(folder_name) :
    file_list = os.listdir(folder_name)
    for file in file_list:
        file_name = folder_name + file
        f = open(file_name,'r')
        lines = f.readlines()
        f.close()
        for idx,line in enumerate(lines):
            if line in ['\n', '\r\n']:
                continue
            elif "SP SPACE " in line:
                continue
            else:
                try:
                    line_list = line.split()
                    if len(line_list) !=4:
                        print(file_name,idx,line)

                except:
                    print(file_name,idx,line)


def find_max_length_sentence(folder_name):
    max_length = 0
    file_list = os.listdir(folder_name)
    for file in file_list:
        file_name = folder_name + file
        f = open(file_name,'r')
        lines = f.readlines()
        f.close()
        temp_len = 0

        for idx,line in enumerate(lines):
            if line in ['\n', '\r\n']:
                if temp_len >110:
                    print(file_name,idx,temp_len)
                if temp_len > max_length:
                    max_length = temp_len
                temp_len = 0
            else:
                temp_len += 1
    return max_length


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



def capital_in_word(word):
    label = False 
    for letter in word[1:]:
        if ord('A') <= ord(letter) <= ord('Z'):
            label = True

    if label == True:
        return np.array([1])
    else:
        return np.array([0])


def label_5(tag):
    one_hot = np.zeros(5)
    if tag.endswith('bc'):
        one_hot[0] =1
    elif tag.endswith('s0'):
        one_hot[1] =1
    elif tag.endswith('s1'):
        one_hot[2] =1
    elif tag.endswith('company'):
        one_hot[3] =1
    elif tag.endswith('O'):
        one_hot[4] =1
    return one_hot

def label_2(tag):
    one_hot = np.zeros(2)
    if tag.endswith('O'):
        one_hot[0] =1
    else:
        one_hot[1] =1
    return one_hot

def label_3(tag):
    one_hot = np.zeros(3)
    if tag.endswith('O'):
        one_hot[0] =1
    elif 's' in tag:
        one_hot[1] =1
    else:
        one_hot[2] =1
    return one_hot



def pickle_file_without_padding(model_file_name,input_file_folder,output_file_name,startfile,endfile,capital=True):
    model = gensim.models.Word2Vec.load(model_file_name)
    train_data = []
    train_label = []
    input_file_names = os.listdir(input_file_folder)
    max_sentence_length = find_max_length_sentence(input_file_folder)
    print(max_sentence_length)

    sentence = []
    sentence_label = []
    for input_file_name in input_file_names[startfile:endfile]:
        f = open(input_file_folder + input_file_name,'r')
        lines = f.readlines() 
        for line in lines:   
            if 'SP SPACE' in line :
                continue      
            if line in ['\n', '\r\n'] :
                if len(sentence) != 0:
                    train_data.append(np.array(sentence))
                    train_label.append(np.array(sentence_label))
                    sentence = []
                    sentence_label = []
                else:
                    continue
                
            else:
                try:
                    assert (len(line.split()) == 4)
                except:
                    continue
                line = line.split()
                word = line[0]
                pos_tag = line[1]
                chunk_tag = line[2]
                label_tag = line[3]
                try:
                    if capital == True:
                        word_embedding = model.wv[word]
                    else:
                        word_embedding = model.wv[word.lower()]
                    pos_embedding = pos(pos_tag)
                    chunk_embedding = chunk(chunk_tag)
                    capital_embedding = capital_in_word(word)
                    label_embedding = label_2(label_tag)
                    embedding = np.append(word_embedding,pos_embedding)
                    embedding = np.append(embedding,chunk_embedding)
                    embedding = np.append(embedding,capital_embedding)
                    sentence.append(embedding)
                    sentence_label.append(label_embedding)
                except:
                    print(line,input_file_name)

    assert(len(train_data) == len(train_label))
    f = open(output_file_name,'wb')
    data = {'train_data': train_data, 'train_label':train_label}
    pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    f.close()


#%%
word2vec_model_path = 'data/word2vec_model/article_model'
pickle_file_without_padding(word2vec_model_path,'data/conll_format_txt/','data/pickle_file/train_sentences_without_padding.pickle',0,850)
pickle_file_without_padding(word2vec_model_path,'data/conll_format_txt/','data/pickle_file/test_sentences_without_padding.pickle',850,934)


