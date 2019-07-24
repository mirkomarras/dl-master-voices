import helpers.generatorutils
import helpers.audioutils
import helpers.coreutils
import tensorflow as tf
import numpy as np
import queue
import time
import os

class Model(object):

    def __init__(self):
        self.graph = None
        self.var2std_epsilon = 0.00001
        self.reuse = False

    def build_model(self, num_classes, nspec, output_dir):
        print("Start building vgg-vector model")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, nspec], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            input_x = tf.layers.batch_normalization(self.input_x, training=is_training, name='bbn0', reuse=self.reuse)

            with tf.variable_scope('conv1'):
                conv1_1 = tf.layers.conv2d(input_x, filters=96, kernel_size=[7, 7], strides=[2, 2], padding='SAME', reuse=self.reuse, name='cc1')
                conv1_1 = tf.layers.batch_normalization(conv1_1, training=is_training, name='bbn1', reuse=self.reuse)
                conv1_1 = tf.nn.relu(conv1_1)
                conv1_1 = tf.layers.max_pooling2d(conv1_1, pool_size=[3, 3], strides=[2, 2], name='mpool1')

            with tf.variable_scope('conv2'):
                conv2_1 = tf.layers.conv2d(conv1_1, filters=256, kernel_size=[5, 5], strides=[2, 2], padding='SAME', reuse=self.reuse, name='cc2')
                conv2_1 = tf.layers.batch_normalization(conv2_1, training=is_training, name='bbn2', reuse=self.reuse)
                conv2_1 = tf.nn.relu(conv2_1)
                conv2_1 = tf.layers.max_pooling2d(conv2_1, pool_size=[3, 3], strides=[2, 2], name='mpool2')

            with tf.variable_scope('conv3'):
                conv3_1 = tf.layers.conv2d(conv2_1, filters=384, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='cc3_1')
                conv3_1 = tf.layers.batch_normalization(conv3_1, training=is_training, name='bbn3', reuse=self.reuse)
                conv3_1 = tf.nn.relu(conv3_1)

                conv3_2 = tf.layers.conv2d(conv3_1, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='cc3_2')
                conv3_2 = tf.layers.batch_normalization(conv3_2, training=is_training, name='bbn4', reuse=self.reuse)
                conv3_2 = tf.nn.relu(conv3_2)

                conv3_3 = tf.layers.conv2d(conv3_2, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='cc3_3')
                conv3_3 = tf.layers.batch_normalization(conv3_3, training=is_training, name='bbn5', reuse=self.reuse)
                conv3_3 = tf.nn.relu(conv3_3)
                conv3_3 = tf.layers.max_pooling2d(conv3_3, pool_size=[5, 3], strides=[3, 2], name='mpool3')

            with tf.variable_scope('conv4'):
                conv4_3 = tf.layers.conv2d(conv3_3, filters=4096, kernel_size=[9, 1], strides=[1, 1], padding='VALID', reuse=self.reuse, name='cc4_1')
                conv4_3 = tf.layers.batch_normalization(conv4_3, training=is_training, name='bbn6', reuse=self.reuse)
                conv4_3 = tf.nn.relu(conv4_3)
                conv4_3 = tf.layers.average_pooling2d(conv4_3, pool_size=[1, conv4_3.shape[2]], strides=[1,1], name='apool4')
                conv4_3 = tf.reduce_mean(conv4_3, axis=[1, 2], name='apool4')

            with tf.variable_scope('fc_audio_vgg'):
                flattened = tf.nn.l2_normalize(conv4_3)
                h = tf.layers.dense(flattened, 1024, reuse=self.reuse, name="scores")

            with tf.variable_scope('output'):
                w = tf.get_variable('w', shape=[1024, num_classes], initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                scores = tf.nn.xw_plus_b(h, w, b, name="scores")

                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=self.input_y)

            self.loss = tf.reduce_mean(losses, name="loss")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            print("Start initializing vgg-vector graph")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir)

        print("VGG-vector building finished")

    @staticmethod
    def save_model(sess, output_dir):
        print("Start saving x-vector graph")
        saver = tf.train.Saver()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = saver.save(sess, os.path.join(output_dir, 'model'))
        with open(os.path.join(output_dir, 'done'), 'wt') as fid:
            fid.write('done')
        print("VGG-vector graph saved in path: %s" % save_path)

    def load_model(self, sess, input_dir):
        print("Start loading vgg-vector graph ...")
        saver = tf.train.import_meta_graph(os.path.join(input_dir, 'model.meta'))
        saver.restore(sess, os.path.join(input_dir, 'model'))
        self.graph = sess.graph
        self.input_x = self.graph.get_tensor_by_name("input_x:0")
        self.input_y = self.graph.get_tensor_by_name("input_y:0")
        self.num_classes = self.input_y.shape[1]
        self.learning_rate = self.graph.get_tensor_by_name("learning_rate:0")
        self.dropout_keep_prob = self.graph.get_tensor_by_name("dropout_keep_prob:0")
        self.phase = self.graph.get_tensor_by_name("phase:0")
        self.loss = self.graph.get_tensor_by_name("loss:0")
        self.optimizer = self.graph.get_operation_by_name("optimizer")
        self.accuracy = self.graph.get_tensor_by_name("accuracy/accuracy:0")
        self.embedding = self.graph.get_tensor_by_name("fc_audio_vgg/scores:0")
        print("VGG-vector graph restored from path: %s" % input_dir)

    def create_one_hot_output_matrix(self, labels):
        minibatch_size = len(labels)
        one_hot_matrix = np.zeros((minibatch_size, self.num_classes), dtype=np.int32)
        for i, lab in enumerate(labels):
            one_hot_matrix[i, lab] = 1
        return one_hot_matrix

    def train_model(self, filterbanks_generator, n_epochs, n_steps_per_epoch, learning_rate, dropout_proportion, print_interval, output_dir):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)) as sess:
            self.load_model(sess, output_dir)

            print("Start training vgg-vector model")
            for epoch in range(n_epochs):
                print('Epoch', epoch, '/', n_epochs)

                minibatch_count = n_steps_per_epoch
                dropout_keep_prob = 1 - dropout_proportion

                total_disk_waiting = 0.0
                total_segments, minibatch_segments = 0, 0
                total_segments_len = 0
                total_gpu_waiting = 0.0
                total_loss, minibatch_loss = 0, 0
                total_accuracy, minibatch_accuracy = 0, 0

                start_time = time.time()
                for minibatch_idx in range(minibatch_count):
                    try:
                        disk_waiting = time.time()
                        batch_data, labels = filterbanks_generator.getitem(minibatch_idx)
                        curr_disk_waiting = time.time() - disk_waiting
                        total_disk_waiting += curr_disk_waiting
                    except queue.Empty:
                        print('Timeout reach when reading the minibatch index %d' % minibatch_idx)
                        continue
                    if batch_data is None:
                        print('batch_data is None for the minibatch index %d' % minibatch_idx)
                        continue

                    batch_labels = self.create_one_hot_output_matrix(labels)

                    total_segments += batch_data.shape[0]
                    total_segments_len += batch_data.shape[1]

                    feed_dict = {self.input_x: batch_data, self.input_y: batch_labels, self.dropout_keep_prob: dropout_keep_prob, self.learning_rate: learning_rate, self.phase: True}
                    gpu_waiting = time.time()
                    _, loss, accuracy = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)
                    curr_gpu_waiting = time.time() - gpu_waiting
                    total_gpu_waiting += curr_gpu_waiting

                    total_loss += loss
                    total_accuracy += accuracy

                    if minibatch_idx % print_interval == 0:
                        print("\rStep %4.0f / %4.0f | AvgLoss: %3.4f | AvgAcc: %3.7f | AvgDiskTime: %3.1f | AvgGPUTime: %3.1f | ElapsedTime: %3.1f" % (minibatch_idx+1, minibatch_count, total_loss / (minibatch_idx+1), total_accuracy / (minibatch_idx+1), curr_disk_waiting, curr_gpu_waiting, time.time() - start_time), end='')

                print()
                Model.save_model(sess, output_dir)

            print("X-vector model trained")

    def extract_embs(self, filterbanks_generator, n_steps_per_epochs, input_dir, embs_output_path, embs_size, min_chunk_size, chunk_size, start_index=0):
        start_time = time.time()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        with tf.Session(config=config) as sess:
            self.load_model(sess, input_dir)

            total_segments = 0
            total_segments_len = 0
            total_gpu_waiting = 0.0
            num_fail = 0
            num_success = 0

            if os.path.exists(embs_output_path):
                print('Loading pre-computed embs till path', start_index)
                embs_matrix = load_npy(embs_output_path)
            else:
                print('Creating embs matrix')
                embs_matrix = np.empty((len(source_audio_paths), embs_size))

            minibatch_count = n_steps_per_epoch

            for minibatch_idx in range(minibatch_count):

                if minibatch_idx >= start_index:
                    mat, label = filterbanks_generator.getitem(minibatch_idx)

                    total_segments += 1
                    num_rows = mat.shape[0]

                    if num_rows == 0:
                        print("Zero-length utterance: '%s'" % path)
                        num_fail += 1
                        continue

                    if num_rows < min_chunk_size:
                        print("Minimum chunk size of %d is greater than the number of rows in utterance: %s" % (min_chunk_size, path))
                        num_fail += 1
                        continue

                    this_chunk_size = chunk_size

                    if num_rows < chunk_size:
                        print("Chunk size of %d is greater than the number of rows in utterance: %s, using chunk size of %d" % (chunk_size, path, num_rows))
                        this_chunk_size = num_rows
                    elif chunk_size == -1:
                        this_chunk_size = num_rows

                    num_chunks = int(np.ceil(num_rows / float(this_chunk_size)))

                    xvector_avg = 0
                    tot_weight = 0.0

                    for chunk_idx in range(num_chunks):
                        offset = min(this_chunk_size, num_rows - chunk_idx * this_chunk_size)
                        if offset < min_chunk_size:
                            continue
                        sub_mat = mat[chunk_idx * this_chunk_size: chunk_idx * this_chunk_size + offset, :]
                        data = np.reshape(sub_mat, (1, sub_mat.shape[0], sub_mat.shape[1]))
                        total_segments_len += sub_mat.shape[0]
                        feed_dict = {self.input_x: data, self.dropout_keep_prob: 1.0, self.phase: False}
                        gpu_waiting = time.time()
                        xvector = sess.run(self.embedding[0], feed_dict=feed_dict)
                        xvector = xvector[0]
                        total_gpu_waiting += time.time() - gpu_waiting
                        tot_weight += offset
                        xvector_avg += offset * xvector

                    xvector_avg /= tot_weight
                    embs_matrix[minibatch_idx,] = xvector_avg
                    num_success += 1

                    save_npy(embs_output_path, embs_matrix)

                    print('\rPath', minibatch_idx, '/', n_steps_per_epochs, 'Fail', num_fail, 'Success', num_success, 'CPUTime', ((time.time() - start_time) / 60.0), 'GPUTime', (total_gpu_waiting / 60.0), end='')