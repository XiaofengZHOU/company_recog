#%%
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np 
import pickle

class LSTM_Model:
    def __init__(self,input_data):
        self.batch_size = 32
        self.n_steps = 94           # timesteps, number of words in one sentence
        self.embedding_size = 312
        self.n_classes = 2
        self.num_layers =2
        self.n_hidden = 512
        self.learning_rate = 0.001
        self.keep_prob = 0.5

        self.tf_model_path = 'data/tf_lstm_model/num_class=2/model/model.ckpt'
        self.sess = tf.Session()
        self.x = tf.placeholder("float", [None, self.n_steps, self.embedding_size])
        self.y = tf.placeholder("float", [None, self.n_steps, self.n_classes])
        self.weights = tf.Variable(tf.random_normal([self.n_hidden*2, self.n_classes],stddev=0.01))
        self.biases =  tf.Variable(tf.random_normal([self.n_classes]))
        self.input_data = input_data

    def monolayer_perceptron(self,x):
        out_layer = tf.matmul(x, self.weights) + self.biases
        return out_layer

    def creat_model(self):
        outputs = self.BiRnn()
        self.pred = self.monolayer_perceptron(outputs)
        self.pred = tf.nn.softmax(self.pred)
        self.pred = tf.reshape(self.pred, [-1, self.n_steps, self.n_classes])

        self.saver = tf.train.Saver()

    def lstm_cell(self):
        # With the latest TensorFlow source code (as of Mar 27, 2017),
        # the BasicLSTMCell will need a reuse parameter which is unfortunately not
        # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
        # an argument check here:
        return tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)

    def attn_cell(self):
        return tf.contrib.rnn.DropoutWrapper(self.lstm_cell(), output_keep_prob=self.keep_prob)


    def BiRnn(self):
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, embedding_size)
        x = tf.unstack(self.x, self.n_steps, 1)
        lstm_fw_cells = rnn.MultiRNNCell([self.attn_cell() for _ in range(self.num_layers)] , state_is_tuple=True)
        lstm_bw_cells = rnn.MultiRNNCell([self.attn_cell() for _ in range(self.num_layers)] , state_is_tuple=True)
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cells, lstm_bw_cells, x,dtype=tf.float32)
        outputs = tf.transpose(tf.stack(outputs), perm=[1, 0, 2])
        outputs = tf.reshape(outputs,[-1,2*self.n_hidden])
        return outputs


    def get_cost(self,prediction,label):
        cross_entropy = label * tf.log(prediction)
        cross_entropy = -1 * tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(label), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
        return tf.reduce_mean(cross_entropy)

    def get_pred(self):
        self.saver.restore(self.sess, self.tf_model_path)
        pr = self.sess.run(self.pred,feed_dict={x: self.input_data})
        return pr

    

#%%
f = open('data/temp/temp.pickle','rb')
data = pickle.load(f)
f.close()
input_data = data['train_data']
text = data['text']

#%%
lstm1 = LSTM_Model(input_data)
lstm1.creat_model()
pr = lstm1.get_pred()

#%%
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph( 'data/tf_lstm_model/num_class=2/model/model.meta')
    new_saver.restore(sess, 'data/tf_lstm_model/num_class=2/model/model.ckpt')
    pred = tf.get_collection("pred")[0]
    x = tf.get_collection("x")[0]
    pr = sess.run(pred,feed_dict={x: input_data})

#%%
res_file = open('data/temp/result.txt','w+')
for idx1,sentence in enumerate(text):
    sent = []
    label_sent=True
    for idx2,word in enumerate(sentence):
        pred_word = np.argmax(pr[idx1][idx2])
        result_word = pr[idx1][idx2]
        line = []
        line.append(word.ljust(20))
        line.append(str(pred_word))
        line.append(str(result_word).ljust(40))
        sent.append(line)
    if label_sent == True:
        for item in sent:
            res_file.write('  '.join(item)+'\n')
        res_file.write('\n')
res_file.close()