import tensorflow as tf 
from data_process import read_corpus
from NetWork import text_cnn
from data_process import load_word2vec
import random
batch_size = 128
epoch_num = 10

seq_length = 70
path_train = "E:\\深度学习\\Dataset\\train.txt"
path_test = "E:\\深度学习\\Dataset\\test.txt"
path_valid = "E:\\深度学习\\Dataset\\validation.txt"
print_per_iters = 50

def make_batch():
    data = read_corpus(path_train)
    random.shuffle(data)
    batch_x = []
    batch_y = []
    for (x,y) in data:
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) >= batch_size:
            yield batch_x, batch_y
            batch_x = []
            batch_y = []
    if len(batch_x) > 0:
        yield batch_x, batch_y


data_X = tf.placeholder(dtype=tf.int32,shape=[None, seq_length])
data_Y = tf.placeholder(dtype=tf.int64,shape=[None])
embeddings = load_word2vec()
# embeddings参数为None,则不适用预训练词向量
loss, accuracy, train_op = text_cnn(data_X, data_Y, embeddings=None)
init = tf.global_variables_initializer() 
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch_num):
        data = make_batch()
        j = 0
        for train_X, train_y in data:
            pre, cost, acc, _ = sess.run([loss, accuracy, train_op], feed_dict={data_X: train_X, data_Y: train_y})
            j += 1
            if 0 == j % print_per_iters:
                print("epoch = %d---------loss=%f---------accuracy=%f" % (i, cost, acc))
        saver.save(sess, "model/textcnn-model.ckpt", global_step=epoch_num)
