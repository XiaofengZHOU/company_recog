#%%
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle


def get_max_length(data):
    temp_len = 0
    max_length = 0
    for sentence in data:
        temp_len = len(sentence)
        if temp_len > max_length:
            max_length =temp_len
    return max_length

def get_data(file_name):
    file =  open(file_name,'rb')
    pickle_data = pickle.load(file)
    data = pickle_data['train_data']
    label = pickle_data['train_label']
    file.close()
    return data,label


train_data, train_label = get_data('data/data_pickle/num_class=2/train_sentences_cap.pickle')
test_data , test_label  = get_data('data/data_pickle/num_class=2/test_sentences_cap.pickle')
test_data2 , test_label2  = get_data('data/data_pickle/num_class=2/test_sentences_cap.pickle')

max_length_train = get_max_length(train_data)
max_length_test  = get_max_length(test_data)
print(max_length_train,max_length_test)
max_length = max_length_train


#%%
batch_size = 32
n_steps = max_length # timesteps, number of words in one sentence
embedding_size = 312
n_classes = 2
num_layers =2
n_hidden = 512
learning_rate = 0.001
keep_prob = 0.5
display_step =10

x = tf.placeholder("float", [None, n_steps, embedding_size])
y = tf.placeholder("float", [None, n_steps, n_classes])
# RNN output node weights and biases

n_hidden = 512
n_hidden_1 = 256
n_hidden_2 =64
weights = {
    'h1': tf.Variable(tf.random_normal([n_hidden*2, n_hidden_1],stddev=0.01)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes],stddev=0.01)),
    'h' : tf.Variable(tf.random_normal([n_hidden*2, n_classes],stddev=0.01))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes])),
    'b' : tf.Variable(tf.random_normal([n_classes]))
}

def monolayer_perceptron(x,weights,biases):
    out_layer = tf.matmul(x, weights['h']) + biases['b']
    return out_layer


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1,0.5)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2,0.8)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def lstm_cell(n_hidden):
    # With the latest TensorFlow source code (as of Mar 27, 2017),
    # the BasicLSTMCell will need a reuse parameter which is unfortunately not
    # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
    # an argument check here:
    return tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)

def attn_cell(n_hidden):
    return tf.contrib.rnn.DropoutWrapper(lstm_cell(n_hidden), output_keep_prob=keep_prob)


def BiRnn(x,n_hidden):
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, embedding_size)
    x = tf.unstack(x, n_steps, 1)
    lstm_fw_cells = rnn.MultiRNNCell([attn_cell(n_hidden) for _ in range(num_layers)] , state_is_tuple=True)
    lstm_bw_cells = rnn.MultiRNNCell([attn_cell(n_hidden) for _ in range(num_layers)] , state_is_tuple=True)
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cells, lstm_bw_cells, x,dtype=tf.float32)
    outputs = tf.transpose(tf.stack(outputs), perm=[1, 0, 2])
    outputs = tf.reshape(outputs,[-1,2*n_hidden])
    return outputs

def get_sentence_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def get_cost(prediction,label):
    cross_entropy = label * tf.log(prediction)
    cross_entropy = -1 * tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(label), reduction_indices=2))
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(cross_entropy)

def get_cost_enforce_entity(prediction,label):
    enforce = np.array([1,100])
    cross_entropy = enforce*label * tf.log(prediction)
    cross_entropy = -1 * tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(label), reduction_indices=2))
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(cross_entropy)

def get_accuracy(pred,label):
    mistakes = tf.equal(tf.argmax(label, 2), tf.argmax(pred, 2))
    mistakes = tf.cast(mistakes, tf.float32)
    mask = tf.sign(tf.reduce_max(tf.abs(label), reduction_indices=2))
    mistakes *= mask
    # Average over actual sequence lengths.
    mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
    mistakes /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(mistakes)

def generate_batch(data,label,length,offset,endset):
    if endset > len(data):
        batch_data = data[offset:]
        batch_label= label[offset:]
        batch_data = batch_data + data[0:endset - len(data)]
        batch_label= batch_label+ label[0:endset - len(data)]
        offset = endset - len(data)
        endset = offset + batch_size
    else:
        batch_data  = data[offset:endset]
        batch_label = label[offset:endset]
        offset = endset
        endset = offset + batch_size
    
    for i in range(len(batch_data)):
        if len(batch_data[i]) < length:
            padding_word = np.array([0 for _ in range(embedding_size)])
            padding_label = np.array([0 for _ in range(n_classes)])
            padding_words = np.array([padding_word   for _ in range(length-len(batch_data[i]))])
            padding_labels = np.array([padding_label for _ in range(length-len(batch_data[i]))])

            batch_data[i]  = np.concatenate( (batch_data[i],padding_words),  axis=0 )
            batch_label[i] = np.concatenate( (batch_label[i],padding_labels),  axis=0 )

    return batch_data,batch_label,offset,endset

def generate_test_data(data,label,length):
    for i in range(len(data)):
        if len(data[i]) < length:
            padding_word = np.array([0 for _ in range(embedding_size)])
            padding_label = np.array([0 for _ in range(n_classes)])
            padding_words = np.array([padding_word   for _ in range(length-len(data[i]))])
            padding_labels = np.array([padding_label for _ in range(length-len(data[i]))])
            data[i]  = np.concatenate( (data[i],padding_words),  axis=0 )
            label[i] = np.concatenate( (label[i],padding_labels),  axis=0 )
    return data,label


def get_entity_accuracy(prediction,label):
    result_mat = np.zeros((5,6))
    for i in range(len(prediction)):
        for j in range(len(label[i])):
            result_label = np.argmax(label[i][j])
            result_pred  = np.argmax(prediction[i][j])
            result_mat[result_label][5] = result_mat[result_label][5] +1
            result_mat[result_label][result_pred] = result_mat[result_label][result_pred] + 1
    return result_mat



#%%
outputs = BiRnn(x,n_hidden)
pred = monolayer_perceptron(outputs,weights,biases)
pred = tf.nn.softmax(pred)
pred = tf.reshape(pred, [-1, n_steps, n_classes])
cost = get_cost(pred,y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = get_accuracy(pred,y)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


#%%
tf_model_path = 'data/tf_lstm_model/num_class=2/model/model.ckpt'
with tf.Session() as sess:
    sess.run(init)
    num_iters = len(train_data)//batch_size
    num_epochs = 20
    print('variable initialized')
    print('num of iters: ', num_iters)
    offset = 0
    endset = 0+batch_size

    for e in range(num_epochs):
        for i in range(num_iters):
            batch_x,batch_y,offset,endset= generate_batch(train_data,train_label,max_length,offset,endset)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            if i%display_step ==0:
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(i) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))
        saver.save(sess,tf_model_path)
        test_data_padding,test_label_padding = generate_test_data(test_data,test_label,max_length)
        pr = sess.run(pred,feed_dict={x: test_data_padding, y: test_label_padding})
        li = get_entity_accuracy(pr,test_label2).astype(int)
        print(e,li)


#%%
for i in range(len(test_data)):
    print(test_data_padding[i].shape,test_label_padding[i].shape)

#%%
tf_model_path = 'data/tf_lstm_model/num_class=2/model/model.ckpt'
with tf.Session() as sess:
    saver.restore(sess, tf_model_path)
    test_data_padding,test_label_padding = generate_test_data(test_data,test_label,max_length)
    pr = sess.run(pred,feed_dict={x: test_data_padding, y: test_label_padding})
    li = get_entity_accuracy(pr,test_label2).astype(int)
    print(li)



#%%

def get_matrix_from_text(model_file_name,conll_text_name,capital=True):
    model = gensim.models.Word2Vec.load(model_file_name)
    f = open(conll_text_name,'r')
    lines = f.readlines()
    f.close()
    text_data = []
    train_data = []
    train_label = []

    text_sentence = []
    sentence = []
    sentence_label = []
    for line in lines:   
        if 'SP SPACE' in line :
            continue      
        if line in ['\n', '\r\n'] :
            if len(sentence) != 0:
                text_data.append(text_sentence)
                train_data.append(np.array(sentence))
                train_label.append(np.array(sentence_label))
                sentence = []
                sentence_label = []
                text_sentence = []
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
                text_sentence.append(word)
                sentence.append(embedding)
                sentence_label.append(label_embedding)
            except:
                print(line)
    return train_data,train_label,text_data

word2vec_model_path = 'data/word2vec_model/article_model_len30_correct_noCapital'
conll_file_path = 'data/review/length30_correct/'
tf_model_path = 'data/tf_model/len30_correct_no_capital/model_length30_noCapital.ckpt'
output_file_path = 'data/tf_model/len30_correct_no_capital/result_review/'

with tf.Session() as sess:
    saver.restore(sess, tf_model_path)
    conll_files = os.listdir(conll_file_path)

    for conll_file_name in conll_files:
        review_data,review_label,review_text = get_matrix_from_text(word2vec_model_path,conll_file_path + conll_file_name)
        review_data_padding,review_label_padding = generate_test_data(review_data,review_label,max_length)
        pr = sess.run(pred,feed_dict={x: review_data_padding, y: review_label_padding})

        res_file = open(output_file_path+conll_file_name+'.txt','w+')
        for idx1,sentence in enumerate(review_text):
            sent = []
            label_sent=False
            for idx2,word in enumerate(sentence):
                label = np.argmax(review_label_padding[idx1][idx2])
                pred_word = np.argmax(pr[idx1][idx2])
                result_word = pr[idx1][idx2]
                line = []
                line.append(word.ljust(20))
                line.append(str(label))
                line.append(str(pred_word))
                line.append(str(result_word).ljust(40))
                if pred_word != label:
                    line.append('*****')
                    label_sent = True
                sent.append(line)
            if label_sent == True:
                for item in sent:
                    res_file.write('  '.join(item)+'\n')
                res_file.write('\n')
        res_file.close()


