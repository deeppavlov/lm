import click
import pickle
import tensorflow as tf
import numpy as np
from corpus import Vocabulary, Corpus
from tf_layers import embedding_layer, cudnn_lstm, stacked_rnn


class LM:
    def __init__(self,
                 vocab_size,
                 emb_dim=256,
                 n_hidden=512,
                 n_layers=1,
                 n_unroll=35,
                 n_hist=15, 
                 model_name='reddit_no_dropout', 
                 gpu=1,
                 bidirectional=True):
        tf.reset_default_graph()
        self.model_name = model_name

        self.tok_ph = tf.placeholder(dtype=tf.int32, shape=[None, n_unroll + 1])
        self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[])
        self.input_dropout = tf.placeholder_with_default(0.5, shape=[])
        self.intra_layer_dropout = tf.placeholder_with_default(0.3, shape=[])
        self.output_dropout = tf.placeholder_with_default(0.5, shape=[])
        self.train_ph = tf.placeholder(dtype=tf.bool, shape=[])
        # self.mask_ph = tf.placeholder_with_default(tf.ones_like(self.tok_ph, dtype=tf.float32), shape=[None, None])
        
        
        self.vocab_size = vocab_size
        self.n_unroll = n_unroll
        self.n_hist = n_hist
        
        # Embeddings
        tok_mat = np.random.randn(self.vocab_size, emb_dim).astype(np.float32) / np.sqrt(emb_dim)
        tok_emb_mat = tf.Variable(tok_mat, name='Embeddings', trainable=True)
        embs = tf.nn.embedding_lookup(tok_emb_mat, self.tok_ph)
        
        # Forward LSTM
        with tf.variable_scope('Forward'):
            units = embs[:, :-1, :]
            units = tf.layers.dropout(units, self.input_dropout, training=self.train_ph)
            for n in range(n_layers):
                with tf.variable_scope('LSTM_' + str(n)):
                    units, _ = cudnn_lstm(units, n_hidden)
                    # units, _ = stacked_rnn(units,
                    #                     n_hidden_list=[n_hidden],
                    #                     cell_type='lstm')
                    print(units)
                    if n != n_layers - 1:
                        units = tf.layers.dropout(units, self.intra_layer_dropout, training=self.train_ph)
            if n_hidden != emb_dim:
                units = tf.layers.dense(units, emb_dim, name='Output_Projection')
            units = tf.layers.dropout(units, self.output_dropout, training=self.train_ph)
            logits_fw = tf.tensordot(units, tok_emb_mat, (2, 1))
            targets = tf.one_hot(self.tok_ph, self.vocab_size)
            fw_loss = tf.losses.softmax_cross_entropy(targets[:, 1:, :], logits_fw[:, :-1, :], reduction=tf.losses.Reduction.NONE)
            fw_loss = 

        self.loss = fw_loss

        if bidirectional:        
            # Backward LSTM
            # Lengths assumed to be equal to n_unroll + n_hist
            lengths =  tf.reduce_sum(tf.ones(tf.shape(self.tok_ph), dtype=tf.int32), 1)
            embs_bw =  tf.reverse_sequence(embs, lengths, seq_axis=1, batch_axis=0)
            with tf.variable_scope('Backward'):
                units = embs_bw[:, :-1, :]
                for n in range(n_layers):
                    with tf.variable_scope('LSTM_' + str(n)):
                        units, _ = cudnn_lstm(units, n_hidden)
                        # units, _ = stacked_rnn(units,
                        #                        n_hidden_list=[n_hidden],
                        #                        cell_type='lstm')
                        units = tf.layers.dropout(units, self.intra_layer_dropout, training=self.train_ph)
                if n_hidden != emb_dim:
                    units = tf.layers.dense(units, emb_dim, name='Output_Projection')
                units = tf.layers.dropout(units, self.output_dropout, training=self.train_ph)
                logits_bw = tf.tensordot(units, tok_emb_mat, (2, 1))
                targets_bw = tf.one_hot(tf.reverse_sequence(self.tok_ph, lengths, seq_axis=1, batch_axis=0), self.vocab_size)
                bw_loss = tf.losses.softmax_cross_entropy(targets_bw[:, n_hist + 1:, :], logits_bw[:, n_hist:, :])
            
            # Train ops
            self.loss = (self.loss + bw_loss) / 2

        tf.summary.scalar('log_loss', self.loss)
        self.summary = tf.summary.merge_all()


        self.train_op = self.get_train_op(self.loss, self.learning_rate_ph, clip_norm=5.0, optimizer_scope_name='Optimizer')

        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(gpu)
        self.sess = tf.Session(config=config)
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
                                      for grad, var in grads_and_vars]
                train_op = opt.apply_gradients(grads_and_vars)
        return train_op
    
    def train(self, corp, batch_size=32, lr=3e-3, every_n=100):
        total_loss = 0
        best_loss = 1e10
        count = 0
        while True:
            for n, x in enumerate(corp.batch_generator(batch_size, self.n_unroll + self.n_hist + 1)):
                loss, summary, _ = self.sess.run([self.loss, self.summary, self.train_op], {self.tok_ph: x,
                                                                     self.learning_rate_ph: lr,
                                                                     self.train_ph: True})
                self.summary_writer.add_summary(summary, count + n)
                total_loss += loss
                if n % every_n == every_n - 1:
                    print(total_loss / every_n)
                    
                    if total_loss / every_n < best_loss:
                        best_loss = total_loss / every_n
                        print('New best loss: {}, model saved'.format(best_loss))
                        self.saver.save(self.sess, 'model/'+ self.model_name + '.ckpt')
                    total_loss = 0
            count += n

@click.command()
@click.option('--lr', default=0.003, help='learning rate')
@click.option('--max-toks', default=100000, help='number of tokens in vocabulary')
@click.option('--gpu', default=1, help='nuber of GPU to use')
@click.option('--every-n', default=100, help='validate every n-th epoch')
@click.option('--bidirectional', default=True, help='whether to build bidirectional or unidirectional LM')
@click.option('--layers', default=1, help='number of layers')
@click.option('--emb-dim', default=128, help='number of layers')
@click.option('--n-hidden', default=256, help='number of hidden units')
@click.option('--model-name', prompt='Model name to save', help='name to save model parameters and summary')
@click.option('--use-rude', default=True, help='use rude words filtered reddit dataset')
def train(lr, max_toks, gpu, every_n, bidirectional, layers, emb_dim, n_hidden, model_name, use_rude):
    if use_rude:
        file_name = 'rude_tokenized_n_min_60.pckl'
    else:
        file_name = 'tokenized_nmin_60.pckl'
    with open(file_name, 'rb') as f:
        corp = pickle.load(f)

    c = Corpus({'train': corp, 'valid': [], 'test': []}, max_tokens=max_toks)
    print('Vocabulary len: {}'.format(len(c.token_dict)))
    with open('model/dict.pckl', 'wb') as f:
        pickle.dump(c.token_dict._i2t, f)'rude_tokenized_n_min_60.pckl'
    net = LM(len(c.token_dict), emb_dim=emb_dim, n_hidden=n_hidden, n_layers=layers, gpu=gpu, bidirectional=bidirectional, model_name=model_name)
    net.train(c, every_n=every_n, lr=lr)


if __name__ == '__main__':
    train()
    
