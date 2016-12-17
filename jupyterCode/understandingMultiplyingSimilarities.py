# coding: utf-8
import os
import pandas as pd
import pickle
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

with open("../model_keys/understandingMultiplying.data", 'rb') as f:
    rank_keys = pickle.load(f)

for file_name in os.listdir("../gensim_models2/understandingMultiplying"):
    similarity_data = []
    if not file_name.endswith(".gsm"):
        continue

    full_path = "../gensim_models2/understandingMultiplying/" + file_name
    problem_type = file_name[:-len(".gsm")].replace('_', ' ')
    model = Word2Vec.load(full_path)

    for seed_rank in model.vocab.keys():
        seed, rank = seed_rank.split('_')
        rank = int(rank)
        original_response = rank_keys[problem_type][seed][rank]
        similarities = [ (other, model.similarity(seed_rank, other)) for other in model.vocab.keys() if
                         not other.startswith(seed)]
        similarities.sort(key=lambda x: x[1], reverse=True)
        # Take 10 most similar responses, but ignore the first item
        # Which is just this same response with a similarity of 1.0
        similarities = similarities[1:11]
        resp_similarities = []
        for sim_seed_rank, _ in similarities:
            sim_seed, sim_rank = sim_seed_rank.split('_')
            sim_rank = int(sim_rank)
            sim_resp = rank_keys[problem_type][sim_seed][sim_rank]
            resp_similarities.append((sim_seed, sim_resp))

        entry = {
            "problem_type" : problem_type,
            "seed" : seed,
            "response" : original_response,
            "most_similar" : repr(resp_similarities),
            "rank" : rank,
        }
        similarity_data.append(entry)

    similarity_df = pd.DataFrame(similarity_data)
    similarity_df.to_csv("../gensim_models2/understandingMultiplying/{}.csv".format(problem_type.replace(' ', '_')),
                         index=False)

for file_name in os.listdir("../gensim_models2/understandingMultiplying"):
    similarity_data = []
    if not file_name.endswith(".gsm"):
        continue

    full_path = "../gensim_models2/understandingMultiplying/" + file_name
    problem_type = file_name[:-len(".gsm")].replace('_', ' ')
    model = Word2Vec.load(full_path)
    vocab = list(model.vocab)
    non_reduced = [ model[word] for word in vocab ]
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(non_reduced)

    tsne_entries = []
    for i in range(len(vocab)):
        seed, rank = vocab[i].split('_')
        rank = int(rank)
        response = rank_keys[problem_type][seed][rank]
        x_coord = reduced[i][0]
        y_coord = reduced[i][1]

        entry = {
            "x" : x_coord,
            "y" : y_coord,
            "seed": seed,
            "response" : response
        }
        tsne_entries.append(entry)
    tsne_df = pd.DataFrame(tsne_entries)
    tsne_df.to_csv("../gensim_models2/understandingMultiplying/{}-tsne.csv".format(problem_type.replace(' ', '_')),
                   index=False)
