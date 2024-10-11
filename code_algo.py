# Import dataset
# pip install surprise
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, SVD, NMF, KNNBasic
from surprise.model_selection import cross_validate
import pandas as pd

# Load the movielens-100k dataset
data = Dataset.load_builtin("ml-100k")

# Function to get RMSE scores from cross-validation
def get_rmse_scores(algo, data):
    results = cross_validate(algo, data, measures=["RMSE"], cv=5, verbose=False)
    return results['test_rmse']  # Return the RMSE scores

# Get RMSE scores for each algorithm
rmse_svd = get_rmse_scores(SVD(), data)
rmse_nmf = get_rmse_scores(NMF(), data)
rmse_knn = get_rmse_scores(KNNBasic(), data)

# Create a DataFrame for seaborn boxplot
df = pd.DataFrame({
    "SVD": rmse_svd,
    "NMF": rmse_nmf,
    "KNNBasic": rmse_knn
})

# Convert the DataFrame into long format for seaborn
df_long = df.melt(var_name="Algorithm", value_name="RMSE")

# Create the box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x="Algorithm", y="RMSE", data=df_long)
plt.title("RMSE Comparison of SVD, NMF, and KNNBasic")
plt.ylabel("RMSE")
plt.xlabel("Algorithm")
plt.grid(True)
plt.show()

