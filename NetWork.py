import tensorflow as tf 
import numpy as np


seq_length = 70
batch_size = 128
vocab_size = 53338 
embedding_dim = 50
drop_keep_prob = 0.5

def text_cnn(input,label, embeddings):
    # 卷积
    # filter_data = tf.Variable(np.random.rand(2,2,3,2),dtype=np.float32)
    # embedding = gensim.models.KeyedVectors.load_word2vec_format("E:\\深度学习\\Dataset\\wiki_word2vec_50.bin",binary=True) 
    print(input.shape)
    '''
    if embeddings is not None:
        embedding = tf.Variable(embeddings, dtype=tf.float32, trainable=True)
    else:
        embedding = tf.Variable(tf.zeros([vocab_size, embedding_dim]),dtype=tf.float32,trainable=True)
    '''

    embedding = tf.get_variable("embedding", [vocab_size, embedding_dim])
    input = tf.nn.embedding_lookup(embedding, input)  # (?, seq_length, embedding_dim)
    
    input = tf.reshape(input, [-1,seq_length,embedding_dim,1])
    kernel = tf.get_variable(name="filter", shape=[3,embedding_dim,1,4], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv = tf.nn.conv2d(input, filter=kernel,strides=(1,1,1,1), padding = "VALID")
    print("conv----shape:",conv.shape)
    
    #conv = tf.layers.conv1d(inputs=input, filters=256, kernel_size=4, strides=1, padding="VALID")    
    # (?, (seq_length - kernel_size)/strides + 1, filters)
    #f1 = tf.reduce_max(conv, axis=[1])   # (?, filters)
    pooling = tf.nn.max_pool(conv, ksize=[1,conv.shape[1],1,1],strides=(1,conv.shape[1],1,1),padding="VALID")
    #biases = tf.Variable(tf.constant(0.0, shape=[4], dtype = tf.float32),name = 'biases')
    #z = tf.nn.bias_add(pooling, biases)
    #pooling = tf.nn.relu(pooling)
    f1 = tf.layers.flatten(pooling)
    #print("f1----shape:",f1.shape)   # (?, 4)
    fc1 = tf.layers.dense(inputs = f1, units=256)
    #fc1 = tf.contrib.layers.dropout(fc1, drop_keep_prob)
    fc1 = tf.nn.relu(fc1)
    output = tf.layers.dense(inputs = fc1, units=2, activation = None)
    precision = tf.argmax(output, -1)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=label))
    accurancy = tf.reduce_sum(tf.cast(tf.equal(label, tf.argmax(output, -1)), dtype=tf.float32)) / batch_size
    train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    return precision,cross_entropy, accurancy, train_op
