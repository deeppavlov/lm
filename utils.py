import numpy as np
import fastText as ft

def assemble_emb_mat(vocab, ft_emb_path):
    model = ft.load_model(ft_emb_path)
    emb_dim = model.get_dimension()
    emb_mat = np.zeros([len(vocab), emb_dim], dtype=np.float32)
    tokens = [token for token in vocab]
    for n, token in enumerate(tokens):
        emb_mat[n] = model.get_word_vector(token)
    return emb_mat
