#%%
import gensim 
import logging
from pprint import pprint
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self,fname):
        self.fname = fname
    def __iter__(self):
        with open(self.fname,'r') as text_file :
            for line in text_file:
                yield line.lower().split()



#%%
sentences = MySentences('data/gensim_word2vec_to_train/gensim.txt') # a memory-friendly iterator
model = gensim.models.Word2Vec(size=300,min_count=1)  # an empty model, no training yet
model.build_vocab(sentences)
model.intersect_word2vec_format('data/word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)
print(model)


#%%
model.train(sentences,total_examples = 933,epochs=10)

#%%
model.save('data/word2vec_model/article_model')

#%%
model.most_similar('cupris',topn=10)