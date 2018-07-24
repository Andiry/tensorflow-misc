import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784 # 输入层节点数，即图片像素
OUTPUT_NODE = 10 # 输出层节点数；输出的是10*1的向量，可参考前文的Example training data label

LAYER1_NODE = 500 #隐藏层节点数

BATCH_SIZE = 100 # 一次训练batch中的数据个数；数字越小（极限为1）则越接近随机梯度下降，越大则越接近梯度下降

LEARNING_RATE_BASE = 0.8     # 基础学习率
LEARNING_RATE_DECAY = 0.99   # 学习率的衰减率
REGULARIZATION_RATE = 0.0001 # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000       # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 若没有提供移动平均类，则直接使用参数当前的取值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    # 若提供了滑动平均类，则首先使用avg_class.average函数计算得出变量的滑动平均值
    # 然后再计算相应的神经网络的前向传播结果
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 模型训练过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 初始化生成隐藏层的参数，这里用truncated_normal而非普通normal，是为了加速训练过程
    # 注：tf.truncated_normal函数的效果是如得到的随机值偏离均值2个标准差以上，则重新随机一次直至在2个标准差以内
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 初始化生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下前向传播的效果，这里滑动平均类为None所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里设定为不可训练变量（trainable=False）
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动拼接衰减率和训练轮数的变量，初始化滑动平均类；在第4章中介绍过给定训练轮数的变量可加快训练早期变量的更新速度
    variale_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 对所有神经网络参数的变量上使用滑动平均（不可训练变量除外）
    variale_averages_op = variale_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播效果
    average_y = inference(x, variale_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵；因为one_hot=True，对于稀疏矩阵可用这个函数来加速交叉熵的计算
    # 【勘误】注意这里原书代码有误
    # 原书是cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))可能跑不通
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算当前batch中所有样例的交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失 = 交叉熵损失 + 正则化损失
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,     # 基础学习率，随着迭代的进行、更新变量时使用的学习率在此基础上递减
        global_step,            # 当前迭代轮次
        mnist.train.num_examples / BATCH_SIZE,  # 做完所有训练需要的总轮次
        LEARNING_RATE_DECAY,    # 学习率衰减速度
        staircase=True)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variale_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # 分别准备验证集和测试集数据
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代训练神经网络
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证集上的结果
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))

# 主程序
def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    train(mnist)

if __name__=='__main__':
    main()
