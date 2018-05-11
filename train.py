import click
import pickle
import tensorflow as tf
import numpy as np
from corpus import Vocabulary, Corpus
from utils import assemble_emb_mat, download
from lm import LM
from pathlib import Path


def download_data():
	emb_url = 'http://lnsigo.mipt.ru/export/embeddings/ft_ru_subs_100.bin'
	data_url = 'http://lnsigo.mipt.ru/export/datasets/subs.pckl'
	fname_emb = 'ft_ru_subs_100.bin'
	fname_data = 'subs.pckl'
	if not Path(fname_emb).exists():
		download(fname_emb, emb_url)
	if not Path(fname_data).exists():
		download(fname_data, data_url)


@click.command()
@click.option('--lr', default=0.003, help='learning rate')
@click.option('--max-toks', default=100000, help='number of tokens in vocabulary')
@click.option('--gpu', default=0, help='nuber of GPU to use')
@click.option('--every-n', default=100, help='validate every n-th epoch')
@click.option('--bidirectional', default=False, help='whether to build bidirectional or unidirectional LM')
@click.option('--layers', default=1, help='number of layers')
@click.option('--emb-dim', default=256, help='number of layers')
@click.option('--n-hidden', default=256, help='number of hidden units')
@click.option('--model-name', prompt='Model name to save', help='name to save model parameters and summary')
def train(lr, max_toks, gpu, every_n, bidirectional, layers, emb_dim, n_hidden, model_name):
    download_data()
    data_file = 'subs.pckl'
    emb_file = 'ft_ru_subs_100.bin' # Fasttext file

    print('Reading dataset')
    with open(data_file, 'rb') as f:
    	# Dataset is a dict with fields 'train', 'test', and 'valid'
    	# each field contains a list of samples:
    	#[['This', 'is', 'sample', '1'], ['This', 'is', 'sample', '2'], ...]
        dataset = pickle.load(f)

    print('Creating corpus')
    c = Corpus(dataset, max_tokens=max_toks)
    print('Vocabulary len: {}'.format(len(c.token_dict)))

    print('Assembling embeddings matrix')
    emb_mat = assemble_emb_mat(c.token_dict, emb_file)

    print('Creating the network')
    net = LM(len(c.token_dict), tok_emb_mat=emb_mat, n_hidden=n_hidden, n_layers=layers, gpu=gpu, bidirectional=bidirectional, model_name=model_name)

    print('Start training')
    net.train(c, every_n=every_n, lr=lr)


if __name__ == '__main__':
    train()
    