#%%
import os 


def generate_gensim_train_text(input_file_folder,output_file_name):
    output_file = open(output_file_name,'w+')
    file_list = os.listdir(input_file_folder)
    for file in file_list:
        sentence = []
        file_name = input_file_folder + file 
        f = open(file_name,'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            try:
                word = line.split()[0]
                sentence.append(word)
            except:
                pass
        output_file.write(' '.join(sentence)+'\n')
    output_file.close()





#%%
input_file_folder = 'data/conll_format_txt/'
output_file_name = 'data/gensim_word2vec_to_train/gensim.txt'
generate_gensim_train_text(input_file_folder,output_file_name)

