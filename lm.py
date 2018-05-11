import tensorflow as tf
from tf_layers import cudnn_lstm


class LM:
    def __init__(self,
                 vocab_size,
                 tok_emb_mat,
                 emb_dim=256,
                 n_hidden=512,
                 n_layers=1,
                 n_unroll=70,
                 model_name='test_model',
                 gpu=1,
                 bidirectional=False,
                 dropout_keep_prob=0.7):
        tf.reset_default_graph()
        self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        self._dropout_ph = tf.placeholder_with_default(1.0, shape=[], name='drop')
        self.tok_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='tok_idxs')
        self.mask_ph = tf.placeholder_with_default(tf.ones_like(self.tok_ph, dtype=tf.float32), shape=[None, None])

        self.model_name = model_name
        self.vocab_size = vocab_size
        self.n_unroll = n_unroll
        self.dropout_keep_prob = dropout_keep_prob
        
        # Embeddings
        emb_mat = tf.Variable(tok_emb_mat, name='Embeddings_Mat', trainable=True)
        embs = tf.nn.embedding_lookup(emb_mat, self.tok_ph)
        
        # Forward LSTM
        with tf.variable_scope('Forward'):
            units = embs[:, :-1, :]
            units = self._variational_dropout(units, self._dropout_ph)
            for n in range(n_layers):
                with tf.variable_scope('LSTM_' + str(n)):
                    units, _ = cudnn_lstm(units, n_hidden)
                    if n != n_layers - 1:
                        units = self._variational_dropout(units, self._dropout_ph)
            if n_hidden != emb_dim:
                units = tf.layers.dense(units, emb_dim, name='Output_Projection')
            units = self._variational_dropout(units, self._dropout_ph)
            logits_fw = tf.tensordot(units, emb_mat, (2, 1))
            targets = tf.one_hot(self.tok_ph, self.vocab_size)

            fw_loss = tf.losses.softmax_cross_entropy(targets[:, 1:, :], logits_fw, reduction=tf.losses.Reduction.NONE)
            fw_loss = self.mask_ph[:, 1:] * fw_loss

        self.loss = fw_loss

        if bidirectional:        
            # Backward LSTM
            # Lengths assumed to be equal to n_unroll + n_hist
            lengths = tf.cast(tf.reduce_sum(self.mask_ph, 1), tf.int32)
            embs_bw = tf.reverse_sequence(embs, lengths, seq_axis=1, batch_axis=0)
            with tf.variable_scope('Backward'):
                units = embs_bw[:, :-1, :]
                for n in range(n_layers):
                    with tf.variable_scope('LSTM_' + str(n)):
                        units, _ = cudnn_lstm(units, n_hidden)
                        if n != n_layers - 1:
                            units = self._variational_dropout(units, self._dropout_ph)
                if n_hidden != emb_dim:
                    units = tf.layers.dense(units, emb_dim, name='Output_Projection')
                units = self._variational_dropout(units, self._dropout_ph)
                logits_bw = tf.tensordot(units, emb_mat, (2, 1))
                targets_bw = tf.one_hot(tf.reverse_sequence(self.tok_ph, lengths, seq_axis=1, batch_axis=0), self.vocab_size)
                bw_loss = tf.losses.softmax_cross_entropy(targets_bw[:, 1:, :], logits_bw, reduction=tf.losses.Reduction.NONE)
                bw_loss = self.mask_ph[:, 1:] * bw_loss
                self.loss = (self.loss + bw_loss) / 2
        self.loss = tf.reduce_sum(self.loss) / tf.reduce_sum(self.mask_ph)

        # Summary
        tf.summary.scalar('log_loss', self.loss)
        self.summary = tf.summary.merge_all()

        # Predictions
        self.pred = tf.argmax(logits_fw, axis=-1)
        if bidirectional:
            self.pred_bw = tf.argmax(tf.reverse_sequence(logits_bw, lengths, seq_axis=1, batch_axis=0), axis=-1)

        # Train ops
        self.train_op = self.get_train_op(self.loss, self.learning_rate_ph, clip_norm=5.0, optimizer_scope_name='Optimizer')

        #  the session
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(gpu)
        self.sess = tf.Session(config=config)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, 'model/reddit_lm.ckpt')        
        self.summary_writer = tf.summary.FileWriter('model/' + self.model_name, self.sess.graph)

    def get_train_op(self,
                     loss,
                     learning_rate,
                     optimizer=None,
                     clip_norm=None,
                     learnable_scopes=None,
                     optimizer_scope_name=None):
        """ Get train operation for given loss

        Args:
            loss: loss, tf tensor or scalar
            learning_rate: scalar or placeholder
            clip_norm: clip gradients norm by clip_norm
            learnable_scopes: which scopes are trainable (None for all)
            optimizer: instance of tf.train.Optimizer, default Adam

        Returns:
            train_op
        """
        if optimizer_scope_name is None:
            opt_scope = tf.variable_scope('Optimizer')
        else:
            opt_scope = tf.variable_scope(optimizer_scope_name)
        with opt_scope:
            if learnable_scopes is None:
                variables_to_train = tf.trainable_variables()
            else:
                variables_to_train = []
                for scope_name in learnable_scopes:
                    for var in tf.trainable_variables():
                        if scope_name in var.name:
                            variables_to_train.append(var)

            if optimizer is None:
                optimizer = tf.train.AdamOptimizer

            # For batch norm it is necessary to update running averages
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                opt = optimizer(learning_rate)
                grads_and_vars = opt.compute_gradients(loss, var_list=variables_to_train)
                if clip_norm is not None:
                    grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var)
                                      for grad, var in grads_and_vars if grad is not None]
                train_op = opt.apply_gradients(grads_and_vars)
        return train_op

    @staticmethod
    def _variational_dropout(units, keep_prob):
        noise_shape = [tf.shape(units)[0], 1, tf.shape(units)[2]]
        return tf.nn.dropout(units, keep_prob, noise_shape)
    
    def train(self, corp, batch_size=32, lr=3e-3, every_n=10000, n_epochs=10):
        total_loss = 0
        best_loss = 1e10
        best_val_loss = 1e10
        count = 0
        for epoch in range(n_epochs):
            print('Epoch {}'.format(epoch))
            for n, (x, mask) in enumerate(corp.batch_generator(batch_size, self.n_unroll)):
                loss, summary, _ = self.sess.run([self.loss, self.summary, self.train_op], {self.tok_ph: x,
                                                                                            self.learning_rate_ph: lr,
                                                                                            self._dropout_ph: self.dropout_keep_prob,
                                                                                            self.mask_ph: mask})
                self.summary_writer.add_summary(summary, count + n)
                total_loss += loss
                if n % every_n == every_n - 1:
                    print(total_loss / every_n)
                    if total_loss / every_n < best_loss:
                        best_loss = total_loss / every_n
                        print('New best loss: {}, model saved'.format(best_loss))
                        self.saver.save(self.sess, 'model/'+ self.model_name + '.ckpt')
                    total_loss = 0
            val_loss = 0
            n_val = 0
            for n, (x, mask) in enumerate(corp.batch_generator(batch_size, self.n_unroll)):

                loss = self.sess.run(self.loss, {self.tok_ph: x,
                                                 self.learning_rate_ph: lr,
                                                 self._dropout_ph: 1.0,
                                                 self.mask_ph: mask})
                val_loss += loss
                n_val += 1
            print('Validation loss: {}'.format(val_loss / n_val))
            if val_loss / n_val < best_loss:
                best_val_loss = val_loss / n_val
                print('New best val loss: {}, model saved'.format(best_loss))
                self.saver.save(self.sess, 'model/'+ self.model_name + '_val.ckpt')
                val_loss = 0
                break
            
            count += n
