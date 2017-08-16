#%%
import tensorflow as tf
import spacy 
import json
import numpy as np
import gensim
import pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
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
                if word.text in [',','?','.','!',';'] and idx<=94:
                    puncs_idx.append(idx)
            try:
                max_punc_idx = max(puncs_idx)
            except:
                continue
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

def generate_gensim_train_sentence(conll_file_name):
    f = open(conll_file_name,'r')
    lines = f.readlines()
    f.close()   
    sentence = [] 
    for line in lines:
        try:
            assert (len(line.split()) == 3)
            word = line.split()[0]
            sentence.append(word)
        except:
            pass
    return ' '.join(sentence)+'\n'
    

def train_gensim_word2vec_with_new_sentence(gensim_model_name,gensim_train_name):
    f = open(gensim_train_name,'r')
    num_sentences = len(f.readlines())
    f.close()
    del f
    if num_sentences >0:
        new_sentence = MySentences(gensim_train_name,capital=True)
        model = gensim.models.Word2Vec.load(gensim_model_name)
        model.build_vocab(new_sentence,update=True)
        model.intersect_word2vec_format('data/word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)    
        model.train(new_sentence,total_examples = num_sentences,epochs=10)
        model.save(gensim_model_name)

def add_all_untrained_articles_to_gensim_file(json_file_name,text_file_name,conll_file_name,gensim_train_name):
    output_file = open(gensim_train_name,'w+')
    raw_articles = get_raw_articles(json_file_name)
    for article in raw_articles:
        if 'tf_companies' not in article.keys():
            text = article['content']
            generate_text_file(text,text_file_name)
            text_conll_format(text_file_name,conll_file_name)
            sentence =  generate_gensim_train_sentence(conll_file_name)
            output_file.write(sentence)
            article['gensim'] = '2'
    output_file.close()
    save_raw_articles(json_file_name,raw_articles)


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

def generate_test_data(data,embedding_size=312,n_classes=2,length=94):
    for i in range(len(data)):
        if len(data[i]) < length:
            padding_word = np.array([0 for _ in range(embedding_size)])
            padding_words = np.array([padding_word   for _ in range(length-len(data[i]))])
            data[i]  = np.concatenate( (data[i],padding_words),  axis=0 )
    return data

def get_company(pr,text):
    company = []
    for idx1,sentence in enumerate(text):
        sent = []
        label_sent=True
        company_name = []
        for idx2,word in enumerate(sentence):         
            pred_word = np.argmax(pr[idx1][idx2])
            if pred_word == 1 and pr[idx1][idx2][1]>0.6:
                company_name.append(word)
            else:
                if len(company_name) != 0:
                    if ' '.join(company_name) not in company:
                        company.append(' '.join(company_name))
                    company_name = []
    return company

def generate_text_file(text,text_file_name):
    f = open(text_file_name,'w+')
    f.write(text)
    f.close()


def get_raw_articles(json_file_name):
    f = open(json_file_name,'r')
    raw_articles = json.load(f)
    f.close()
    return raw_articles

def save_raw_articles(json_file_name,raw_articles):
    f = open(json_file_name,'w')
    json.dump(raw_articles, f,indent = 2)
    f.close()
       
#%%
if __name__ == "__main__":
    gensim_model_name = 'data/word2vec_model/article_model'
    text_file_name = 'data/temp/temp.txt'
    conll_file_name = 'data/temp/temp_conll.txt'
    gensim_train_name = 'data/temp/gensim_train.txt'
    json_file_name = 'data/raw_articles.json'
    add_all_untrained_articles_to_gensim_file(json_file_name,text_file_name,conll_file_name,gensim_train_name)
    train_gensim_word2vec_with_new_sentence(gensim_model_name,gensim_train_name)

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph( 'data/tf_lstm_model/num_class=2/model/model.meta')
        new_saver.restore(sess, 'data/tf_lstm_model/num_class=2/model/model.ckpt')
        pred = tf.get_collection("pred")[0]
        x = tf.get_collection("x")[0]

        raw_articles = get_raw_articles(json_file_name)
        for idx,article in enumerate(raw_articles):
            if 'tf_companies' in article.keys():
                print('skip')
                continue
            else:
                text = article['content']
                generate_text_file(text,text_file_name)
                text_conll_format(text_file_name,conll_file_name)
                input_data,text_data = get_word2vec_embeddings(gensim_model_name,conll_file_name)
                input_data = generate_test_data(input_data)   
                
                try:
                    pr = sess.run(pred,feed_dict={x: input_data})
                    company = get_company(pr,text_data)
                    article['tf_companies'] = company
                    print(company)
                except:
                    pass

            if idx%10 == 0:
                save_raw_articles(json_file_name,raw_articles)

