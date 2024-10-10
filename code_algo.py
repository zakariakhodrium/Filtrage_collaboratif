#%%
from surprise import Dataset, SVD, NMF, KNNBasic
from surprise.model_selection import cross_validate
import pandas as pd
import time

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin("ml-100k")
start_time = time.time()

# We'll use the famous SVD algorithm.
# Run 5-fold cross-validation and print results
# %%
algo = SVD()
cross_validate(algo, data, measures= ["RMSE"], cv=5, verbose=True)
# %%
algo2= NMF()
cross_validate(algo2, data, measures=["RMSE"], cv=5, verbose=True)
# %%
algo3 = KNNBasic()
cross_validate(algo3, data, measures = ["RMSE"], cv = 5, verbose= True)
elapsed_time = time.time() - start_time
print(f"Elapsed time for parallel execution: {elapsed_time:.2f} seconds")

# %%
from surprise import Dataset, SVD, NMF, KNNBasic
from surprise.model_selection import cross_validate
from joblib import Parallel, delayed
import pandas as pd
import time

# Load the movielens-100k dataset
data = Dataset.load_builtin("ml-100k")

# Define a function for cross-validation
def run_cross_validation(algo):
    return cross_validate(algo, data, measures=["RMSE"], cv=5, verbose=False)

# List of algorithms to evaluate
algorithms = [SVD(), NMF(), KNNBasic()]

# Measure time for parallel execution
start_time = time.time()

# Run cross-validation in parallel for all algorithms
results = Parallel(n_jobs=-1)(delayed(run_cross_validation)(algo) for algo in algorithms)

# Calculate elapsed time
elapsed_time = time.time() - start_time

# Print results
for algo, result in zip(algorithms, results):
    print(f"Results for {algo.__class__.__name__}:")
    print(result)
    print("\n")

print(f"Elapsed time for parallel execution: {elapsed_time:.2f} seconds")
