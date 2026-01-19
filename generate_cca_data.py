
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

def generate_synthetic_cca_data(n_samples=500, n_features_x=10, n_features_y=10, output_file='cca_results.csv'):
    np.random.seed(42)
    
    # Latent variable
    z = np.random.normal(0, 1, n_samples)
    
    # X variables: Gene expression related
    # Gene_1 is highly correlated with z
    X = np.zeros((n_samples, n_features_x))
    X_names = [f'Gene_{i+1}' for i in range(n_features_x)]
    
    # Gene_1 = APOE (Let's give it a real name for the agent to find)
    X_names[0] = 'APOE'
    X[:, 0] = z + np.random.normal(0, 0.3, n_samples)
    
    # Other genes are random noise
    for i in range(1, n_features_x):
        X[:, i] = np.random.normal(0, 1, n_samples)
        
    # Y variables: Brain regions
    # Hippocampus is highly correlated with z
    Y = np.zeros((n_samples, n_features_y))
    Y_names = [f'Region_{i+1}' for i in range(n_features_y)]
    
    # Region_1 = Hippocampus
    Y_names[0] = 'Hippocampus'
    Y[:, 0] = z + np.random.normal(0, 0.3, n_samples)
    
    # Region_2 = Amygdala (moderate correlation)
    Y_names[1] = 'Amygdala'
    Y[:, 1] = 0.5 * z + np.random.normal(0, 0.8, n_samples)

    # Other regions are random noise
    for i in range(2, n_features_y):
        Y[:, i] = np.random.normal(0, 1, n_samples)

    # Perform CCA
    cca = CCA(n_components=1)
    cca.fit(X, Y)
    
    # Get weights (loadings)
    x_weights = cca.x_weights_.flatten()
    y_weights = cca.y_weights_.flatten()
    
    # Create DataFrame for results
    # We want to present the top contributors to the first canonical variate
    
    # Combine into a single list or separate lists?
    # Let's create a format where we list Component 1 loadings
    
    df_x = pd.DataFrame({'Variable': X_names, 'Type': 'Gene', 'Weight': x_weights})
    df_y = pd.DataFrame({'Variable': Y_names, 'Type': 'Brain_Region', 'Weight': y_weights})
    
    df_results = pd.concat([df_x, df_y]).sort_values(by='Weight', key=abs, ascending=False)
    
    print("Top contributors to Canonical Variate 1:")
    print(df_results.head())
    
    df_results.to_csv(output_file, index=False)
    print(f"CCA results saved to {output_file}")

if __name__ == "__main__":
    generate_synthetic_cca_data()
