import os
import time
import logging as log
import torch

log.warning("Sheer Loading transformer_wrapper")

from sentence_transformers import SentenceTransformer

model=None

def check_models(modeldir='/home/sheer/.cache/torch/sentence_transformers/'):
    if not os.path.exists(modeldir):
        return []
    return [fl for fl in os.listdir(modeldir)]

def check_gpu_access():
    return torch.cuda.is_available()


def load_model(name='bert-base-nli-mean-tokens', reload_=False):
    global model
    t0=time.time()
    if model is None or reload_:
        log.warning("Loading model %s"%name)
        model = SentenceTransformer(name)
    t1=time.time()
    return t1-t0, str(model)

def gen_sent_features(sent_list, batch_size = 50):
    load_model()
    # use only one thread so we rely purely on dask for parallelization
    torch.set_num_threads(1)
    t0=time.time()
    sentence_embeddings = model.encode(sent_list, batch_size=batch_size)
    t1=time.time()
    log.warning("took %.3fs for gen_sent_features with %d sents (%d threads)"%(t1-t0,
                len(sent_list), torch.get_num_threads()))
    return sentence_embeddings

