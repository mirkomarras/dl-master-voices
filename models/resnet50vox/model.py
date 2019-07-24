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

    def __get_variable(self, name, shape, initializer, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)

    def batch_norm_wrapper(self, inputs, is_training, decay=0.99, epsilon=1e-3, name_prefix=''):
        gamma = self.__get_variable(name_prefix + 'gamma', inputs.get_shape()[-1], tf.constant_initializer(1.0))
        beta = self.__get_variable(name_prefix + 'beta', inputs.get_shape()[-1], tf.constant_initializer(0.0))
        pop_mean = self.__get_variable(name_prefix + 'mean', inputs.get_shape()[-1], tf.constant_initializer(0.0), trainable=False)
        pop_var = self.__get_variable(name_prefix + 'variance', inputs.get_shape()[-1], tf.constant_initializer(1.0), trainable=False)
        axis = list(range(len(inputs.get_shape()) - 1))

        def in_training():
            batch_mean, batch_var = tf.nn.moments(inputs, axis)
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

        def in_evaluation():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)

        return tf.cond(is_training, lambda: in_training(), lambda: in_evaluation())

    def build_model(self, num_classes, nfilt, output_dir):
        layer_sizes = [512, 512, 512, 512, 3 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]

        print("Start building x-vector model")

        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.num_classes = num_classes

            # placeholder for parameter
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.phase = tf.placeholder(tf.bool, name="phase")

            # Placeholders for regular data
            self.input_x = tf.placeholder(tf.float32, [None, None, nfilt], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            h = self.input_x

            # Frame level information Layer
            prev_dim = nfilt
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    kernel_shape = [kernel_size, prev_dim, layer_size]
                    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")

                    conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h = tf.nn.bias_add(conv, b)

                    # Apply nonlinearity and BN
                    h = tf.nn.relu(h, name="relu")
                    h = self.batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = layer_size

                    # Apply dropout
                    if i != len(kernel_sizes) - 1:
                        with tf.name_scope("dropout-%s" % i):
                            h = tf.nn.dropout(h, self.dropout_keep_prob)

            # Statistic pooling
            tf_mean, tf_var = tf.nn.moments(h, 1)
            h = tf.concat([tf_mean, tf.sqrt(tf_var + self.var2std_epsilon)], 1)
            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):

                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")

                    h = tf.nn.xw_plus_b(h, w, b, name="scores")

                    h = tf.nn.relu(h, name="relu")
                    h = self.batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim
                    if i != len(embedding_sizes) - 1:
                        with tf.name_scope("dropout-%s" % i):
                            h = tf.nn.dropout(h, self.dropout_keep_prob)

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, num_classes], initializer=tf.contrib.layers.xavier_initializer())
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
            print("Start initializing x-vector graph")
            sess.run(tf.global_variables_initializer())
            Model.save_model(sess, output_dir)

        print("X-vector building finished")

    @staticmethod
    def save_model(sess, output_dir):
        print("Start saving x-vector graph")
        saver = tf.train.Saver()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = saver.save(sess, os.path.join(output_dir, 'model'))
        with open(os.path.join(output_dir, 'done'), 'wt') as fid:
            fid.write('done')
        print("X-vector graph saved in path: %s" % save_path)

    def load_model(self, sess, input_dir):
        print("Start loading x-vector graph ...")
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
        self.embedding = [None] * 2  # TODO make this more general
        self.embedding[0] = self.graph.get_tensor_by_name("embed_layer-0/scores:0")
        self.embedding[1] = self.graph.get_tensor_by_name("embed_layer-1/scores:0")
        print("X-vector graph restored from path: %s" % input_dir)

    def create_one_hot_output_matrix(self, labels):
        minibatch_size = len(labels)
        one_hot_matrix = np.zeros((minibatch_size, self.num_classes), dtype=np.int32)
        for i, lab in enumerate(labels):
            one_hot_matrix[i, lab] = 1
        return one_hot_matrix

    def print_models_params(self, input_dir):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            self.load_model(sess, input_dir)
            print('\n\nThe x-vector components are:\n')
            for v in self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                print(v.name)
            print('\n')

    def get_models_weights(self, input_dir):
        import h5py
        h5file = os.path.join(input_dir, 'model.h5')
        if os.path.exists(h5file):
            name2weights = {}

            def add2weights(name, mat):
                if not isinstance(mat, h5py.Group):
                    name2weights[name] = mat.value

            with h5py.File(h5file, 'r') as hf:
                hf.visititems(add2weights)
            return name2weights

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            self.load_model(sess, input_dir)
            name2weights = {}
            for v in self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                name2weights[v.name] = sess.run(v)
                print('%s  shape: %s' % (v.name, str(name2weights[v.name].shape)))
            for i in range(5):
                for scope_name in ("frame_level_info_layer-%s" % i, "embed_layer-%s" % i):
                    for var_name in ("mean", "variance"):
                        name = '%s/%s:0' % (scope_name, var_name)
                        try:
                            name2weights[name] = sess.run(self.graph.get_tensor_by_name(name))
                            print('%s  shape: %s' % (name, str(name2weights[name].shape)))
                        except:
                            pass
            with h5py.File(h5file, 'w') as hf:
                for name, mat in name2weights.iteritems():
                    hf.create_dataset(name, data=mat.astype(np.float32))
            return name2weights

    def train_model(self, filterbanks_generator, n_epochs, n_steps_per_epoch, learning_rate, dropout_proportion, print_interval, output_dir):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)) as sess:
            self.load_model(sess, output_dir)

            print("Start training x-vector model")
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