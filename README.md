# Info 290 Final Project

Topic: Analyzing Misconceptions on Khan Academy

Team Members: Jack Kolb, Gao Xian Peh, Jong Ha Lee

## Jupyter Notebooks

The bulk of our data preprocessing, data analysis, and model training was performed within Jupyter notebooks. The Python code from all of the notebooks used in our project are included in this repository. Below is a rough summary of each notebook's code.

### `understandingMultiplyingFractionsFinal.py`
This set of code was used to clean and analyze the data we had from Khan Academy's "Understanding Multiplying Fractions by Fractions" exercise. It performs the following high-level steps.

1. Extract the relevant fields for each student's response from the raw data set: timestamp, correctness (as determined by Khan Academy), problem type, random seed, student identifier, and actual response content.

2. Extract the student's actual answer from each response's raw content, which was in a messy JSON format.

3. Clean up problems with the "problem type" field. This mainly involved throwing out nonsense problem types, manually reassigning problem types of some responses where we had performed manual verification, and resolving problem type "0" to the true problem type based on other occurrences of the same seed.

4. Compute the frequency of each response within in seed group, which in turn is within each problem type group. The results are then emmitted as CSV files for future manual analysis.

5. Train a skip-gram model using the responses (arranged chronologially and grouped by student ID to form "sentences") and save this model to a file for future processing in a separate notebook. We performed some basic hyper-parameter optimization in which we attempted to maximize the total similarity of response pairs that are known to be the result of the same underlying misconception, as revealed by our manual analysis.
