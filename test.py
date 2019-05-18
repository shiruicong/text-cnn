import tensorflow as tf 
from data_process import read_corpus
from NetWork import text_cnn
from data_process import load_word2vec
import random
batch_size = 128
epoch_num = 10

seq_length = 70
path_test = "E:\\深度学习\\Dataset\\test.txt"

print_per_iters = 50

def make_batch():
    data = read_corpus(path_test)
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
precision, loss, accuracy, train_op = text_cnn(data_X, data_Y, embeddings=None)
init = tf.global_variables_initializer()
path_model = tf.train.latest_checkpoint("./model") 
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,path_model)
    data = make_batch()
    j = 0
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    avg_f = 0
    for train_X, train_y in data:
        p= sess.run(precision, feed_dict={data_X: train_X})
        for i in range(len(train_y)):
            if p[i]==train_y[i] and p[i]==1:   #预测为1，实际为1
                TP = TP + 1
            if p[i]==train_y[i] and p[i]==0:   #预测为0，实际为0
                TN = TN + 1
            if p[i]!=train_y[i] and p[i]==0:   #预测为0,但实际是1
                FN = FN + 1
            if p[i]!=train_y[i] and p[i]==1:   #预测为1，但实际是0 
                FP = FP + 1
    acc = TP/(TP+FP)
    recall = TP/(TP+FN)
    f =( 2 * acc * recall)/(acc+recall)
    print("TP---%d,TN---%d,FN---%d,FP---%d" % (TP,TN,FN,FP))
    print("accurancy-----%f, recall----%f" % (acc,recall))
    print("f1值=======%f" % (f))
    # saver.save(sess, "model/textcnn-model.ckpt", global_step=epoch_num)