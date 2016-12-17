# coding: utf-8
import itertools
import json
import os
import pickle
import pandas as pd
from gensim.models import Word2Vec

raw_data = pd.read_csv("../data/multiplying-fractions-by-fractions-word-problems_all.log")

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


# Remove problem types that are very infrequent
included_prob_types = []
for problem_type in frequency_data["problem_type"].unique():
    relevant_data = frequency_data[ frequency_data["problem_type"] == problem_type]
    if len(relevant_data) > 500:
        included_prob_types.append(problem_type)
frequency_data = frequency_data[ frequency_data["problem_type"].isin(included_prob_types) ]

# Remove seeds that only map to the erroneous problem type "0"
# There are only 1879 responses corresponding to these seeds
degenerate_seeds = []
for seed in frequency_data["seed"].unique():
    relevant_data = frequency_data[ frequency_data["seed"] == seed ]
    problem_types = relevant_data["problem_type"].unique()
    if len([problem_types]) == 1 and problem_types[0] == '0':
        degenerate_seeds.append(seed)
frequency_data = frequency_data[ ~frequency_data["seed"].isin(degenerate_seeds) ]

# We manually verified that the following seeds all fall under the corresponding problem types
seed_remap = {
    "x199b68d6" : "Type 2: MX x FRAC",
    "xa27dedee" : "Type 2: MX x FRAC",
    "xf47f4b42" : "Type 1: FRAC x FRAC",
    "xb2575ae9" : "Type 1: FRAC x FRAC",
    "x35d4b973" : "Type 2: MX x FRAC",
    "xfaa08261" : "Type 1: FRAC x FRAC"
}
def manualSeedRemap(row):
    new_prob_type = seed_remap.get(row["seed"], row["problem_type"])
    row["problem_type"] = new_prob_type
    return row
frequency_data = frequency_data.apply(manualSeedRemap, axis=1)

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

print("Number of responses: {}".format(len(frequency_data)))
print("Number of problem types: {}".format(len(frequency_data["problem_type"].unique())))
print("Number of seeds: {}".format(len(frequency_data["seed"].unique())))
print("Number of users: {}".format(len(frequency_data["student"].unique())))

for seed in frequency_data["seed"].unique():
    relevant_data = frequency_data[ frequency_data["seed"] == seed ]
    problem_types = relevant_data["problem_type"].unique()
    if len(problem_types) > 1:
        print("Error for seed {}: {}".format(seed, problem_types))
    assert len(problem_types) == 1

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

with open("../model_keys/factionWordProblems.data", 'wb') as f:
    pickle.dump(rank_keys, f)

frequency_output = pd.DataFrame(frequencies)
frequency_output.sort_values(by=["problem_type", "seed", "frequency"], inplace=True, ascending=[True, True, False])
frequency_output.to_csv("../frequency_results/fractionWordProblems3.csv", index=False)

frequency_data["time_done"] = pd.to_datetime(frequency_data["time_done"])
frequency_data.head()

def optimize_model(problem_type, sentences, window_sizes, dimensionalities, cluster_dir):
    expected_clusters = []
    file_name = cluster_dir + os.sep + problem_type.replace(':', '') + ".csv"
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

frequency_data.dropna(inplace=True)
for problem_type, problem_type_group in frequency_data.groupby("problem_type"):
    if problem_type.startswith("TYPE 3: TABLES"):
        continue
    print("{}: {}".format(problem_type, len(problem_type_group)))
    all_sentences = []
    for student, student_responses in problem_type_group.groupby("student"):
        student_sentences = []
        incorrect_responses = student_responses[ ~student_responses["correct"] ]
        for _, row in incorrect_responses.sort_values("time_done").iterrows():
            seed = row["seed"]
            response = row["attempts"]
            rank = frequency_rank_dict[problem_type][seed][response]
            else:
                student_sentences.append("{}_{}".format(seed, rank))
        all_sentences.append(student_sentences)

    window_sizes = [4, 1, 2, 3, 5, 6]
    dimensionalities = [100, 25, 50, 75, 150, 200]
    optimal_model = optimize_model(problem_type, all_sentences, window_sizes, dimensionalities, "../gaoClusters")
    output_file = "../gensim_models2/wordProblems/{}.gsm".format(problem_type.replace(' ', '_'))
    optimal_model.save(output_file)
