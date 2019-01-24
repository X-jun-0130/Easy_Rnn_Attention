import tensorflow as tf
from Parameters import Parameters as pm
from data_processing import *

class Rnn_Attention(object):

    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, pm.num_classes], name='input_y')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='seq_length')
        self.keep_pro = tf.placeholder(tf.float32, name='drop_out')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.Rnn_attention()

    def Rnn_attention(self):
        with tf.name_scope('Cell'):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(pm.hidden_dim)
            Cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, self.keep_pro)

            cell_bw = tf.contrib.rnn.BasicLSTMCell(pm.hidden_dim)
            Cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, self.keep_pro)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.get_variable('embedding', shape=[pm.vocab_size, pm.embedding_dim],
                                             initializer=tf.constant_initializer(pm.pre_trianing))
            self.embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope('biRNN'):

            output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=Cell_fw, cell_bw=Cell_bw, inputs=self.embedding_input,
                                                        sequence_length=self.seq_length, dtype=tf.float32)
            output = tf.concat(output, 2) #[batch_size, seq_length, 2*hidden_dim]

        with tf.name_scope('attention'):
            u_list = []
            seq_size = output.shape[1].value
            hidden_size = output.shape[2].value #[2*hidden_dim]
            attention_w = tf.Variable(tf.truncated_normal([hidden_size, pm.attention_size], stddev=0.1), name='attention_w')
            attention_u = tf.Variable(tf.truncated_normal([pm.attention_size, 1], stddev=0.1), name='attention_u')
            attention_b = tf.Variable(tf.constant(0.1, shape=[pm.attention_size]), name='attention_b')
            for t in range(seq_size):
                #u_t:[1,attention]
                u_t = tf.tanh(tf.matmul(output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
                u = tf.matmul(u_t, attention_u)
                u_list.append(u)
            logit = tf.concat(u_list, axis=1)
            #u[seq_size:attention_z]
            weights = tf.nn.softmax(logit, name='attention_weights')
            #weight:[seq_size:1]
            out_final = tf.reduce_sum(output * tf.reshape(weights, [-1, seq_size, 1]), 1)
            #out_final:[batch,hidden_size]

        with tf.name_scope('dropout'):
            self.out_drop = tf.nn.dropout(out_final, keep_prob=self.keep_pro)

        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([hidden_size, pm.num_classes], stddev=0.1), name='w')
            b = tf.Variable(tf.zeros([pm.num_classes]), name='b')
            self.logits = tf.matmul(self.out_drop, w) + b
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            # 退化学习率 learning_rate = lr*(0.9**(global_step/10);staircase=True表示每decay_steps更新梯度
            # learning_rate = tf.train.exponential_decay(self.config.lr, global_step=self.global_step,
            # decay_steps=10, decay_rate=self.config.lr_decay, staircase=True)
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            # self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step) #global_step 自动+1
            # no.2
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1

        with tf.name_scope('accuracy'):
            correct = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    def feed_data(self, x_batch, y_batch, seq_length, keep_pro):
        feed_dict = {self.input_x: x_batch,
                    self.input_y: y_batch,
                    self.seq_length: seq_length,
                    self.keep_pro: keep_pro}

        return feed_dict

    def evaluate(self, sess, x, y):
        batch_test = batch_iter(x, y, batch_size=64)
        for x_batch, y_batch in batch_test:
            seq_len = sequence(x_batch)
            feed_dict = self.feed_data(x_batch, y_batch, seq_len, 1.0)
            test_loss, test_accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        return test_loss, test_accuracy









