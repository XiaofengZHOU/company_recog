#%%
import spacy
from bs4 import BeautifulSoup
from pprint import pprint
import numpy as np 
import os 
nlp = spacy.load('en')


#%%
def get_max_sentence(sentences):
    max_length = 0 
    for sentence in sentences:
        max_length = max(max_length,len(sentence))
    return max_length


def get_annotations(main_text,annotations_tag):
    annotations_list = []
    for annotation_tag in annotations_tag:
        annotation  = []
        if annotation_tag.has_attr('type'):
            type = annotation_tag['type']
        else:
            type = None 
        if type == 'Company':
            startnode = int( annotation_tag['startnode'] )
            endnode = int( annotation_tag['endnode'] )
            try:
                feature_name = annotation_tag.find('name').getText()
                feature_value= annotation_tag.find('value').getText()
            except:
                feature_name = None
                feature_value = None
                pass
            
            annotation.append(type)
            annotation.append(main_text[startnode:endnode])
            annotation.append( (startnode,endnode) )
            annotation.append(feature_name)
            annotation.append(feature_value)
            annotations_list.append(annotation)

    return annotations_list

def get_entity(annotation):
    entity = ''
    if annotation[3] == None or annotation[3] =='':
        entity = 'company'
    elif 'bc' in annotation[3].lower():
        entity = 'bc'
    elif 'vc' in annotation[3].lower():
        entity = 'vc'
    elif 's' in annotation[3].lower() :
        if annotation[4] == '1':
            entity = 's1'
        else:
            entity = 's0'
    else:
        entity = 'company'
    return entity

def word_annotation(word,annotations_list,f):
    word_text = word.text
    line = ' ' + word.tag_ +  ' ' + word.pos_  + ' '
    word_startnode = word.idx
    word_endnode   = word.idx + len(word_text)
    word_range = np.arange(word_startnode,word_endnode)
    word_label = False
    for annotation in annotations_list:
        entity = get_entity(annotation)
        start = annotation[2][0]
        end = annotation[2][1]
        feature_range = np.arange(start,end)
        intersection = list( set(word_range).intersection(feature_range) )
        intersection.sort()

        if len(intersection) == 0:
            continue
        else:
            word_label = True
            word_no_anno1 = main_text[ word_range[0]:intersection[0] ]
            word_anno = main_text[ intersection[0]:intersection[-1]+1 ] 
            word_no_anno2 = main_text[ intersection[-1]:word_range[-1] ]
            if len(word_no_anno1) !=0 :
                f.write(word_no_anno1 + line  + 'O' + '\n')
            f.write(word_anno+' '+ line  + entity +'\n')
            if len(word_no_anno2) !=0 :
                f.write(word_no_anno2 + line  + 'O' + '\n')
            break
    if word_label == False:
        f.write(word_text + line + 'O' + '\n')

def write_to_file(multi_sentences,annotations_list,f):
    for sentence in multi_sentences:
        for word in sentence:
            word_annotation(word,annotations_list,f)
    f.write('\n')


def convert_to_conll(main_text,annotations_list,output_file_name,multi_sentences_length):
    f = open(output_file_name,'w+')
    doc = nlp(main_text)
    sents_list = [sent for sent in doc.sents]
    sents = doc.sents
    multi_sentences= []
    num_words = 0
    total = 0 
    for sent in sents:
        if len(sent) > multi_sentences_length: 
            write_to_file(multi_sentences,annotations_list,f)
            multi_sentences= []
            total = total + num_words
            num_words = 0
            
            multi_sentences.append(sent)
            write_to_file(multi_sentences,annotations_list,f)
            multi_sentences= []
            total = total + len(sent)
            

        elif num_words + len(sent) > multi_sentences_length:
            write_to_file(multi_sentences,annotations_list,f)
            multi_sentences= []
            total = total + num_words

            num_words = len(sent)
            multi_sentences= []
            multi_sentences.append(sent)

            if sent.text == sents_list[-1].text:
                print('zzzzzzzzzzzzzzzzzzzzzzzzzz')
                write_to_file(multi_sentences,annotations_list,f)
                total = total + num_words
        else:
            multi_sentences.append(sent) 
            num_words = num_words + len(sent)
            if sent.text == sents_list[-1].text:
                print('ggggggggggggggggggggggg')
                write_to_file(multi_sentences,annotations_list,f)
                total = total + num_words
    print(output_file_name, ' ', len(doc), ' ',total)
    f.close()





#%%
# multi_sentences_length = 100 
# file_path = 'data/annotations/'
# file_list = os.listdir(file_path)
# for file in file_list:
#     xml_file_name = file_path + file
#     soup = BeautifulSoup(open(xml_file_name),"html5lib")
#     annotations_tag = soup.find_all('annotation')
#     main_text_tag = soup.find('textwithnodes')
#     main_text = main_text_tag.getText()   
#     annotations_list = get_annotations(main_text,annotations_tag) 
#     convert_to_conll(main_text,annotations_list,'data/conll_format_txt/'+file,multi_sentences_length)



#%%
multi_sentences_length = 30
file_path = 'data/annotations/'
xml_file_name = file_path + '902'
soup = BeautifulSoup(open(xml_file_name),"html5lib")
annotations_tag = soup.find_all('annotation')
main_text_tag = soup.find('textwithnodes')
main_text = main_text_tag.getText() 
annotations_list = get_annotations(main_text,annotations_tag) 
pprint(annotations_list)
convert_to_conll(main_text,annotations_list,'data/conll_format_txt/length30/'+'902',multi_sentences_length)








