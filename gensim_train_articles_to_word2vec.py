#%%
import gensim 
import logging
from pprint import pprint
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
sentences = MySentences('data/gensim_sentence_train/gensim.txt',capital=True) # a memory-friendly iterator
model = gensim.models.Word2Vec(size=300,min_count=1)  # an empty model, no training yet
model.build_vocab(sentences)
model.intersect_word2vec_format('data/word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)
print(model)
model.train(sentences,total_examples = 933,epochs=10)
model.save('data/word2vec_model/article_model')

#%%
sentences = MySentences('data/gensim_sentence_train/gensim.txt',capital=False) # a memory-friendly iterator
model = gensim.models.Word2Vec(size=300,min_count=1)  # an empty model, no training yet
model.build_vocab(sentences)
model.intersect_word2vec_format('data/word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)
print(model)
model.train(sentences,total_examples = 933,epochs=10)
model.save('data/word2vec_model/article_model_no_capital')

#%%
model.most_similar('york',topn=10)