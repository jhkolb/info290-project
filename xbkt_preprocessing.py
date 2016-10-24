import pandas as pd

def convertCorrect(correct):
    if correct == "True":
        return 1
    else:
        return 0

print("Reading CSV")
raw_data = pd.read_csv("/research/khananswers/data/recognizing_fractions_0.5_all.log")
print("Sorting by Timestamp")
raw_data['time_done'] = pd.to_datetime(raw_data['time_done'])
relevant_data = raw_data[["time_done", "sha1", "correct", "problem_type"]]

skill_groups = relevant_data.groupby("problem_type")
for name, group in skill_groups:
    print("Processing skill " + name)
    srted = group.sort_values(["time_done", "sha1"])
    responses = []
    start_points = []
    lengths = []
    running_start = 1
    for _, problems in srted.groupby("sha1"):
        responses += problems["correct"].apply(convertCorrect).toList()
        lengths.append(problems.shape[0])
        start_points.append(running_start)
        running_start += problems.shape[0]

    with open(name + "_data.csv", 'w') as f:
        f.write(' '.join([str(int(response)) for response in responses]))
    with open(name + "_starts.csv", 'w') as f:
        f.write(' '.joint([str(start_point) for start_point in start_points]))
    with open(name + "_lengths.csv", 'w') as f:
        f.write(' '.joint([str(length) for length in lengths]))
