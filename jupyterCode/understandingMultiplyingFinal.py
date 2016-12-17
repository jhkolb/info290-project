# coding: utf-8
import itertools
import json
import os
import pickle
import pandas as pd
from gensim.models import Word2Vec

raw_data = pd.read_csv("../data/understanding-multiplying-fractions-by-fractions_all.log")

frequency_data = raw_data[["time_done", "correct", "attempts", "problem_type", "seed"]].copy()
frequency_data["student"] = raw_data["f0_"].apply(lambda s: s.split(':')[1])
frequency_data.dropna(inplace=True)
frequency_data.head()

def extract_answer(s):
    try:
        data = json.loads(s)
    except ValueError:
        print("Failed to parse JSON")
    while isinstance(data, list):
        data = data[0]

    response = None
    if isinstance(data, dict):
        response = data.get("currentValue")
        if response is None:
            response = data.get("value")
    return response

frequency_data["attempts"] = frequency_data["attempts"].apply(extract_answer)
frequency_data.dropna(inplace=True)

# Remove seeds that only map to the erroneous problem type "0"
# There are only 6 responses corresponding to these seeds
frequency_data = frequency_data[ ~frequency_data["seed"].isin(["xde7749ca", "x0d7b4037", "x5db5e4f65eb032f4"]) ]

# There are very few responses for these problem types, so we throw them out
low_freq_types = ["Less than 1", "Type 1: EXPRESSION", "Type 1: VISUALS, RECTANGLES", "Type 2: VISUAL, SQUARES"]
frequency_data = frequency_data[ ~frequency_data["problem_type"].isin(low_freq_types) ]

# We have manually verified that these are really all 'TYPE 3: TAPE DIAGRAMS' problems
bad_seeds = [
    "x17d8457616b6982a",
    "x56b78fe6b2fe2eb2",
    "x86794260f2432645",
    "x8db3b67f9410b6d3",
    "xdabc8fb8f292822b",
    "xf1a40b7b0dc93543",
    "xfae7378c5e4cff08",
]

def fixProblemTypes(row):
    if row["seed"] in bad_seeds:
        row["problem_type"] = "Type 3: TAPE DIAGRAMS"
    elif row["seed"] == "x1376ff8c449cc2d5":
        row["problem_type"] = "Type 1"
    return row
frequency_data = frequency_data.apply(fixProblemTypes, axis=1)

seed_mappings = {}
for _, row in frequency_data.iterrows():
    if row["problem_type"] != "0":
        seed_mappings[row["seed"]] = row["problem_type"]

def replacePTZeros(row):
    if row["problem_type"] == "0":
        row["problem_type"] = seed_mappings[row["seed"]]
    return row
frequency_data = frequency_data.apply(replacePTZeros, axis=1)
frequency_data.head()

for seed in frequency_data["seed"].unique():
    relevant_data = frequency_data[ frequency_data["seed"] == seed ]
    problem_types = relevant_data["problem_type"].unique()
    assert len(problem_types) == 1


print("Number of responses: {}".format(len(frequency_data)))
print("Number of problem types: {}".format(len(frequency_data["problem_type"].unique())))
print("Number of seeds: {}".format(len(frequency_data["seed"].unique())))
print("Number of users: {}".format(len(frequency_data["student"].unique())))

frequencies = []
frequency_rank_dict = {}
rank_keys = {}
response_keys = {}
for problem_type, problem_type_group in frequency_data.groupby("problem_type"):
    frequency_rank_dict[problem_type] = {}
    rank_keys[problem_type] = {}
    for seed, seed_group in problem_type_group.groupby("seed"):
        frequency_rank_dict[problem_type][seed] = {}
        rank_keys[problem_type][seed] = {}
        response_keys[seed] = {}
        incorrect_responses = seed_group[ ~seed_group["correct"] ]
        seed_frequencies = []
        for response, response_group in incorrect_responses.groupby("attempts"):
            frequency_entry = {
                "problem_type" : problem_type,
                "seed": seed,
                "response": response,
                "frequency": len(response_group)
            }
            frequencies.append(frequency_entry)
            seed_frequencies.append(frequency_entry)
        seed_frequencies.sort(key=lambda x: x["frequency"], reverse=True)
        for i, seed_freq in enumerate(seed_frequencies):
            response = seed_freq["response"]
            frequency_rank_dict[problem_type][seed][response] = (i + 1)
            rank_keys[problem_type][seed][i + 1] = response
            response_keys[seed][response] = (i + 1)

with open("../model_keys/understandingMultiplying.data", 'wb') as f:
    pickle.dump(rank_keys, f)

frequency_output = pd.DataFrame(frequencies)
frequency_output.sort_values(by=["problem_type", "seed", "frequency"], inplace=True, ascending=[True, True, False])
frequency_output.to_csv("../frequency_results/understandingMultiplying3.csv", index=False)

frequency_data["time_done"] = pd.to_datetime(frequency_data["time_done"])
frequency_data.head()

def optimize_model(problem_type, sentences, window_sizes, dimensionalities, cluster_dir):
    expected_clusters = []
    file_name = cluster_dir + os.sep + problem_type + "_seedresps.csv"
    if not os.path.exists(file_name):
        print("No constraints found for problem type {}, using default params".format(problem_type))
        model = Word2Vec(sentences, min_count=5, size=100, window=4, sg=1)
        return model


    cluster_data = pd.read_csv(cluster_dir + os.sep + file_name)
    expected_clusters = []
    for _, miscon_group in cluster_data.groupby("expla.group"):
        if len(miscon_group) < 2:
            continue
        cluster = []
        for _, row in miscon_group.iterrows():
            response = row["response"]
            seed = row["seed"]
            response_ranks = response_keys.get(seed)
            if response_ranks is None:
                print("Missing seed {}".format(seed))
            else:
                rank = response_ranks[response]
                word = "{}_{}".format(seed, rank)
                cluster.append(word)
        expected_clusters.append(cluster)

    all_combos = itertools.product(window_sizes, dimensionalities)
    first_window, first_dimensions = next(all_combos)
    first_model = Word2Vec(sentences, min_count=5, size=first_dimensions, window=first_window, sg=1)
    present_clusters = [ [ word for word in cluster if word in first_model.vocab ]
                         for cluster in expected_clusters ]
    sufficient_clusters = [ cluster for cluster in present_clusters if len(cluster) >= 2 ]
    if len(sufficient_clusters) == 0:
        print("Not enough similarity constraints, using default parameters ({}, {})".format(
              first_window, first_dimensions))
        return first_model

    max_similarity = 0.0
    for cluster in sufficient_clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                max_similarity = first_model.similarity(cluster[i], cluster[j])
    best_model = first_model
    best_dimensions = first_dimensions
    best_window = first_window
    for window_size, dimensions in all_combos:
        model = Word2Vec(sentences, min_count=5, size=dimensions, window=window_size, sg=1)
        similarity = 0.0
        for cluster in sufficient_clusters:
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    similarity += model.similarity(cluster[i], cluster[j])
        if similarity > max_similarity:
            best_model = model
            best_window = window_size
            best_dimensions = dimensions
            max_similarity = similarity

    print("Best window size: {}".format(best_window))
    print("Best dimensionality: {}".format(best_dimensions))
    return best_model

for problem_type, problem_type_group in frequency_data.groupby("problem_type"):
    print("{}: {}".format(problem_type, len(problem_type_group)))
    all_sentences = []
    for student, student_responses in problem_type_group.groupby("student"):
        student_sentences = []
        incorrect_responses = student_responses[ ~student_responses["correct"] ]
        for _, row in incorrect_responses.sort_values("time_done").iterrows():
            seed = row["seed"]
            response = row["attempts"]
            rank = frequency_rank_dict[problem_type][seed][response]
            student_sentences.append("{}_{}".format(seed, rank))
        all_sentences.append(student_sentences)

    window_sizes = [4, 1, 2, 3, 5, 6]
    dimensionalities = [100, 25, 50, 75, 150, 200]
    optimal_model = optimize_model(problem_type, all_sentences, window_sizes, dimensionalities, "../jongClusters")
    output_file = "../gensim_models2/understandingMultiplying/{}.gsm".format(problem_type.replace(' ', '_'))
    optimal_model.save(output_file)
