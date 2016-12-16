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

### `fractionWordProblemsFinal.py`
This code is nearly identical to that of `understandingMultiplyingFractionsFinal.py`, except that it is applied to the "Multiplying Fractions by Fractions Word Problems" exercise instead. Thus, there is some slightly different code to clean up inconsistencies in the "problem type" labels for some of the responses, but all other aspects are essentially the same.

### `understandingMultiplyingSimilarities.py`
This notebook reads in the external skip-gram model files (one for each problem type) generated by `understandingMultiplyingFractionsFinal.py`It then uses these models to perform two major tasks.

1. For each unique response, find the ten most similar responses. These similarities are then written to a CSV file for manual analysis and hopefully verification that the associations surfaced by the model reflect shared misconceptions.

2. Perform dimensionality reduction on the vector embeddings representing each response. We used the t-SNE module from `scikit-learn` to reduce all vectors to two dimensions. The reduced representations are then saved to a CSV file for future plotting and cluster analysis.

### `wordProblemSimilarities.py`
This code identifies similarities and performs t-SNE dimensionality reduction, just like `understandingMultiplyingSimilarities.py`, but for the "Multiplying Fractions by Fractions Word Problems" dataset.

## R Code
