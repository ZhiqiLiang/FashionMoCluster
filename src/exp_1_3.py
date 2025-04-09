import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score, 
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
import torch
from matplotlib.colors import ListedColormap
import time
import gc
import psutil
import os

class EnhancedClusteringExperiment:
    def __init__(self, test_data_path, features_paths, output_dir="outputs"):
        """
        Initialize the clustering experiment with data paths
        
        Parameters:
        -----------
        test_data_path : str
            Path to the test data CSV file
        features_paths : dict
            Dictionary of feature paths with keys as feature names
        output_dir : str
            Directory to save output files
        """
        self.test_data_path = test_data_path
        self.features_paths = features_paths
        self.output_dir = output_dir
        self.df_test = None
        self.feature_dataframes = {}
        self.feature_arrays = {}
        self.original_data = None
        self.true_labels = None
        self.label_names = {
            0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 
            4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 
            8: 'Bag', 9: 'Ankle boot'
        }
        self.colors = [
            'red', 'green', 'blue', 'purple', 'magenta', 
            'yellow', 'cyan', 'maroon', 'teal', 'black'
        ]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load and prepare the data"""
        # Load test data
        self.df_test = pd.read_csv(self.test_data_path)
        self.true_labels = self.df_test['label'].to_numpy()
        
        # Extract original image data for PCA-based methods
        if 'label' in self.df_test.columns:
            self.original_data = self.df_test.drop(['label'], axis=1).to_numpy()
        else:
            self.original_data = self.df_test.to_numpy()
        
        # Load feature vectors from each path
        for feature_name, path in self.features_paths.items():
            df = pd.read_csv(path)
            self.feature_dataframes[feature_name] = df
            # Extract features (assuming label is in the dataframe)
            if 'label' in df.columns:
                self.feature_arrays[feature_name] = df.drop(['label'], axis=1).to_numpy()
            else:
                self.feature_arrays[feature_name] = df.to_numpy()
        
        return self
        
    def preprocess_data(self):
        """Preprocess the data for clustering"""
        # Scale original data
        scaler = StandardScaler()
        self.original_data = scaler.fit_transform(self.original_data)
        
        # Scale each feature array
        for feature_name in self.feature_arrays:
            scaler = StandardScaler()
            self.feature_arrays[feature_name] = scaler.fit_transform(self.feature_arrays[feature_name])
        
        return self
    
    def apply_pca(self, data, n_components=50):
        """
        Apply PCA dimensionality reduction
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data to reduce dimensions
        n_components : int
            Number of PCA components to keep
            
        Returns:
        --------
        reduced_data : numpy.ndarray
            Dimensionality-reduced data
        """
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"PCA with {n_components} components explains {explained_variance:.2%} of variance")
        return reduced_data
    
    def run_kmeans(self, data, n_clusters=10, track_resources=True):
        """
        Run K-means clustering on the given data
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data to cluster
        n_clusters : int
            Number of clusters
        track_resources : bool
            Whether to track memory and time resources
            
        Returns:
        --------
        labels : numpy.ndarray
            Cluster labels
        metrics : dict
            Performance metrics (time and memory if tracked)
        """
        process = psutil.Process()
        metrics = {}
        
        if track_resources:
            # Disable garbage collection to measure memory accurately
            gc.disable()
            start_memory = process.memory_info().rss / (1024 ** 2)
            start_time = time.time()
        
        # Initialize and fit K-means model
        model = KMeans(
            init="k-means++", 
            n_clusters=n_clusters, 
            n_init=35,
            random_state=42
        )
        
        model.fit(data)
        labels = model.labels_
        
        if track_resources:
            # Calculate resource usage
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024 ** 2)
            metrics['time'] = end_time - start_time
            metrics['memory'] = end_memory - start_memory
            gc.enable()
        
        return labels, metrics
    
    def run_spectral_clustering(self, data, n_clusters=10, track_resources=True):
        """
        Run Spectral clustering on the given data
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data to cluster
        n_clusters : int
            Number of clusters
        track_resources : bool
            Whether to track memory and time resources
            
        Returns:
        --------
        labels : numpy.ndarray
            Cluster labels
        metrics : dict
            Performance metrics (time and memory if tracked)
        """
        process = psutil.Process()
        metrics = {}
        
        if track_resources:
            # Disable garbage collection to measure memory accurately
            gc.disable()
            start_memory = process.memory_info().rss / (1024 ** 2)
            start_time = time.time()
        
        # Initialize and fit Spectral clustering model
        # Note: Using 'nearest_neighbors' for affinity as it works well with high-dimensional data
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=10,
            random_state=42,
            n_init=35
        )
        
        labels = model.fit_predict(data)
        
        if track_resources:
            # Calculate resource usage
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024 ** 2)
            metrics['time'] = end_time - start_time
            metrics['memory'] = end_memory - start_memory
            gc.enable()
        
        return labels, metrics
    
    def evaluate_clustering(self, data, true_labels, cluster_labels, method_name):
        """
        Evaluate clustering performance using various metrics
        
        Parameters:
        -----------
        data : numpy.ndarray
            Original data used for clustering
        true_labels : numpy.ndarray
            Ground truth labels
        cluster_labels : numpy.ndarray
            Predicted cluster labels
        method_name : str
            Name of the clustering method for reporting
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Supervised metrics (require true labels)
        metrics["ARI"] = adjusted_rand_score(true_labels, cluster_labels)
        metrics["NMI"] = normalized_mutual_info_score(true_labels, cluster_labels)
        
        # Unsupervised metrics (require data)
        metrics["Silhouette"] = silhouette_score(data, cluster_labels)
        metrics["CHI"] = calinski_harabasz_score(data, cluster_labels)
        metrics["CHI_normalized"] = torch.log1p(torch.tensor(metrics["CHI"])) / 10.0
        metrics["DBI"] = davies_bouldin_score(data, cluster_labels)
        
        # Print results
        print(f"Method: {method_name}")
        print(f"  Adjusted Rand Index (ARI): {metrics['ARI']:.4f}")
        print(f"  Normalized Mutual Information (NMI): {metrics['NMI']:.4f}")
        print(f"  Silhouette Coefficient: {metrics['Silhouette']:.4f}")
        print(f"  Calinski-Harabasz Index (normalized): {metrics['CHI_normalized']:.4f}")
        print(f"  Davies-Bouldin Index: {metrics['DBI']:.4f}")
        print("============================================================")
        
        return metrics
    
    def get_cluster_indexes(self, labels):
        """Get indexes of samples for each cluster"""
        num_clusters = len(np.unique(labels))
        cluster_indexes = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            cluster_indexes[label].append(i)
        return cluster_indexes
    
    def visualize_2d_clusters(self, data, labels, title, use_true_labels=False):
        """
        Create a 2D scatter plot of clustered data
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data to visualize (will be PCA-reduced if dimensions > 2)
        labels : numpy.ndarray
            Cluster labels (or true labels if use_true_labels=True)
        title : str
            Plot title
        use_true_labels : bool
            Whether to use true labels instead of cluster labels
        """
        # Apply PCA for dimensionality reduction to 2D
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
        else:
            data_2d = data
        
        # Create figure and save it
        plt.figure(figsize=(12, 10))
        
        if use_true_labels:
            # Use true labels and corresponding class names
            unique_labels = np.unique(labels)
            cmap = ListedColormap(self.colors[:len(unique_labels)])
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(
                    data_2d[mask, 0], 
                    data_2d[mask, 1],
                    c=[self.colors[i % len(self.colors)]],
                    label=self.label_names.get(label, f"Class {label}"),
                    alpha=0.7,
                    s=30
                )
        else:
            # Use cluster labels
            unique_clusters = np.unique(labels)
            for i, cluster in enumerate(unique_clusters):
                mask = labels == cluster
                plt.scatter(
                    data_2d[mask, 0], 
                    data_2d[mask, 1],
                    c=[self.colors[i % len(self.colors)]],
                    label=f"Cluster {cluster}",
                    alpha=0.7,
                    s=30
                )
        
        plt.title(title, fontsize=16)
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{title.replace(' ', '_').replace(':', '')}.png"))
        plt.close()
    
    def export_cluster_distribution_to_csv(self, labels, method_name):
        """
        Generate a CSV file showing the distribution of true labels within each cluster
        
        Parameters:
        -----------
        labels : numpy.ndarray
            Cluster labels from clustering algorithm
        method_name : str
            Name of the clustering method used
        
        Returns:
        --------
        csv_path : str
            Path to the generated CSV file
        """
        # Get cluster indices
        cluster_indexes = self.get_cluster_indexes(labels)
        
        # Create a DataFrame to store the distribution data
        distribution_data = []
        
        # Process each cluster
        for cluster_id in range(len(cluster_indexes)):
            # If cluster is empty, continue to next cluster
            if len(cluster_indexes[cluster_id]) == 0:
                continue
                
            # Get true labels for this cluster
            cluster_true_labels = self.true_labels[cluster_indexes[cluster_id]]
            
            # Count occurrences of each label in this cluster
            label_count = {}
            for label in cluster_true_labels:
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1
            
            # Total number of items in this cluster
            total_items = len(cluster_true_labels)
            
            # Add data for each label in this cluster
            for label, count in label_count.items():
                # Calculate frequency
                frequency = count / total_items
                
                # Add row to distribution data
                distribution_data.append({
                    'Cluster_ID': cluster_id,
                    'Label_ID': label,
                    'Label_Name': self.label_names[label] if hasattr(self, 'label_names') and label in self.label_names else f"Label_{label}",
                    'Count': count,
                    'Frequency': frequency,
                    'Cluster_Total': total_items
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(distribution_data)
        
        # Define file path
        file_name = f"cluster_distribution_{method_name.replace(' ', '_').replace('-', '_').replace(':', '')}.csv"
        csv_path = os.path.join(self.output_dir, file_name)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        
        print(f"Cluster distribution for {method_name} saved to {csv_path}")
        
        return csv_path

    def visualize_cluster_distribution(self, labels, title):
        """
        Visualize the distribution of true classes within each cluster
        
        Parameters:
        -----------
        labels : numpy.ndarray
            Cluster labels
        title : str
            Plot title
        """
        cluster_indexes = self.get_cluster_indexes(labels)
        
        plt.figure(figsize=(18, 20))
        for i in range(len(cluster_indexes)):
            plt.subplot(5, 2, i+1)
            
            # Count true labels within this cluster
            label_count = {}
            for label in self.true_labels[cluster_indexes[i]]:
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1
            
            if label_count:  # Only create plot if cluster has members
                plt.bar(range(len(label_count)), list(label_count.values()), align='center')
                plt.title(f'Cluster {i}')
                plt.xticks(
                    range(len(label_count)), 
                    [self.label_names[key] for key in label_count.keys()],
                    rotation=45, 
                    ha='right'
                )
                plt.tight_layout()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(self.output_dir, f"{title.replace(' ', '_').replace(':', '')}.png"))
        plt.close()
    
    def run_experiment(self, methods=None, pca_components=128):
        """
        Run clustering experiments on selected feature sets and methods
        
        Parameters:
        -----------
        methods : list or None
            List of method names to run. If None, run all methods.
            Possible values: ['MoCo Pretraining Embeddings', 'MoCo K-means Finetuning Embeddings', 'MoCo Spectral Finetuning Embeddings',
                            'PCA-KMeans', 'PCA-Spectral']
        pca_components : int
            Number of components to use for PCA-based methods
            
        Returns:
        --------
        results : dict
            Dictionary of results with metrics and labels
        comparison_df : pandas.DataFrame
            DataFrame with comparison metrics
        """
        # Define all available methods
        all_methods = list(self.features_paths.keys()) + ['PCA-KMeans', 'PCA-Spectral']
        
        # If methods is None, use all methods
        if methods is None:
            methods = all_methods
        else:
            # Ensure all specified methods are valid
            for method in methods:
                if method not in all_methods:
                    raise ValueError(f"Invalid method: {method}. Available methods: {all_methods}")
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Create table for comparison
        comparison_metrics = {
            'Method': [],
            'ARI': [],
            'NMI': [],
            'Silhouette': [],
            'CHI': [],
            'DBI': [],
            'Time (s)': [],
            'Memory (MB)': []
        }
        
        results = {}
        
        # Run clustering on each selected feature set
        for method in methods:
            if method == 'MoCo Pretraining Embeddings':
                # Use both K-Means and Spectral Clustering for this method
                feature_name = method
                feature_data = self.feature_arrays[feature_name]
                
                # Run K-Means
                print(f"\nRunning K-means on {feature_name}...")
                kmeans_labels, kmeans_resource_metrics = self.run_kmeans(feature_data)
                kmeans_eval_metrics = self.evaluate_clustering(
                    feature_data, self.true_labels, kmeans_labels, f"K-means on {feature_name}"
                )
                self.export_cluster_distribution_to_csv(kmeans_labels, f"K-means on {feature_name}")
                combined_kmeans_metrics = {**kmeans_resource_metrics, **kmeans_eval_metrics}
                results[f"K-means on {feature_name}"] = {
                    'labels': kmeans_labels,
                    'metrics': combined_kmeans_metrics
                }
                
                # Add K-Means results to comparison table
                comparison_metrics['Method'].append(f"K-means on {feature_name}")
                comparison_metrics['ARI'].append(kmeans_eval_metrics['ARI'])
                comparison_metrics['NMI'].append(kmeans_eval_metrics['NMI'])
                comparison_metrics['Silhouette'].append(kmeans_eval_metrics['Silhouette'])
                comparison_metrics['CHI'].append(kmeans_eval_metrics['CHI'])
                comparison_metrics['DBI'].append(kmeans_eval_metrics['DBI'])
                comparison_metrics['Time (s)'].append(kmeans_resource_metrics['time'])
                comparison_metrics['Memory (MB)'].append(kmeans_resource_metrics['memory'])

                
                # Visualize K-Means results
                print(f"\nVisualizing distributions for K-means on {feature_name}...")
                self.visualize_cluster_distribution(
                    kmeans_labels, f"Class Distribution in Clusters - K-means on {feature_name}"
                )
                self.visualize_2d_clusters(
                    feature_data, kmeans_labels, f"2D Cluster Visualization - K-means on {feature_name}"
                )
                self.visualize_2d_clusters(
                    feature_data, 
                    self.true_labels, 
                    f"2D Visualization with True Labels - K-means on {feature_name}", 
                    use_true_labels=True
                )
                
                # Run Spectral Clustering
                print(f"\nRunning Spectral Clustering on {feature_name}...")
                spectral_labels, spectral_resource_metrics = self.run_spectral_clustering(feature_data)
                spectral_eval_metrics = self.evaluate_clustering(
                    feature_data, self.true_labels, spectral_labels, f"Spectral Clustering on {feature_name}"
                )
                self.export_cluster_distribution_to_csv(spectral_labels, f"Spectral Clustering on {feature_name}")
                combined_spectral_metrics = {**spectral_resource_metrics, **spectral_eval_metrics}
                results[f"Spectral Clustering on {feature_name}"] = {
                    'labels': spectral_labels,
                    'metrics': combined_spectral_metrics
                }
                
                # Add Spectral Clustering results to comparison table
                comparison_metrics['Method'].append(f"Spectral Clustering on {feature_name}")
                comparison_metrics['ARI'].append(spectral_eval_metrics['ARI'])
                comparison_metrics['NMI'].append(spectral_eval_metrics['NMI'])
                comparison_metrics['Silhouette'].append(spectral_eval_metrics['Silhouette'])
                comparison_metrics['CHI'].append(spectral_eval_metrics['CHI'])
                comparison_metrics['DBI'].append(spectral_eval_metrics['DBI'])
                comparison_metrics['Time (s)'].append(spectral_resource_metrics['time'])
                comparison_metrics['Memory (MB)'].append(spectral_resource_metrics['memory'])
                
                # Visualize Spectral Clustering results
                print(f"\nVisualizing distributions for Spectral Clustering on {feature_name}...")
                self.visualize_cluster_distribution(
                    spectral_labels, f"Class Distribution in Clusters - Spectral Clustering on {feature_name}"
                )
                self.visualize_2d_clusters(
                    feature_data, spectral_labels, f"2D Cluster Visualization - Spectral Clustering on {feature_name}"
                )
                self.visualize_2d_clusters(
                    feature_data, 
                    self.true_labels, 
                    f"2D Visualization with True Labels - Spectral Clustering on {feature_name}", 
                    use_true_labels=True
                )

            elif method == 'MoCo K-means Finetuning Embeddings':
                # Only K-Means for this method
                feature_name = method
                feature_data = self.feature_arrays[feature_name]
                method_name = f"K-means on {feature_name}"
                print(f"\nRunning {method_name}...")
                
                # Run K-Means
                labels, resource_metrics = self.run_kmeans(feature_data)
                eval_metrics = self.evaluate_clustering(
                    feature_data, self.true_labels, labels, method_name
                )
                self.export_cluster_distribution_to_csv(labels, f"K-means on {feature_name}")
                combined_metrics = {**resource_metrics, **eval_metrics}
                results[method_name] = {
                    'labels': labels,
                    'metrics': combined_metrics
                }
                
                # Add to comparison table
                comparison_metrics['Method'].append(method_name)
                comparison_metrics['ARI'].append(eval_metrics['ARI'])
                comparison_metrics['NMI'].append(eval_metrics['NMI'])
                comparison_metrics['Silhouette'].append(eval_metrics['Silhouette'])
                comparison_metrics['CHI'].append(eval_metrics['CHI'])
                comparison_metrics['DBI'].append(eval_metrics['DBI'])
                comparison_metrics['Time (s)'].append(resource_metrics['time'])
                comparison_metrics['Memory (MB)'].append(resource_metrics['memory'])
                
                # Visualize results
                print(f"\nVisualizing distributions for {method_name}...")
                self.visualize_cluster_distribution(labels, f"Class Distribution in Clusters - {method_name}")
                self.visualize_2d_clusters(feature_data, labels, f"2D Cluster Visualization - {method_name}")
                self.visualize_2d_clusters(
                    feature_data, 
                    self.true_labels, 
                    f"2D Visualization with True Labels - {method_name}", 
                    use_true_labels=True
                )

            elif method == 'MoCo Spectral Finetuning Embeddings':
                # Only Spectral Clustering for this method
                feature_name = method
                feature_data = self.feature_arrays[feature_name]
                method_name = f"Spectral Clustering on {feature_name}"
                print(f"\nRunning {method_name}...")
                
                # Run Spectral Clustering
                labels, resource_metrics = self.run_spectral_clustering(feature_data)
                eval_metrics = self.evaluate_clustering(
                    feature_data, self.true_labels, labels, method_name
                )
                self.export_cluster_distribution_to_csv(labels, f"Spectral Clustering on {feature_name}")
                combined_metrics = {**resource_metrics, **eval_metrics}
                results[method_name] = {
                    'labels': labels,
                    'metrics': combined_metrics
                }
                
                # Add to comparison table
                comparison_metrics['Method'].append(method_name)
                comparison_metrics['ARI'].append(eval_metrics['ARI'])
                comparison_metrics['NMI'].append(eval_metrics['NMI'])
                comparison_metrics['Silhouette'].append(eval_metrics['Silhouette'])
                comparison_metrics['CHI'].append(eval_metrics['CHI'])
                comparison_metrics['DBI'].append(eval_metrics['DBI'])
                comparison_metrics['Time (s)'].append(resource_metrics['time'])
                comparison_metrics['Memory (MB)'].append(resource_metrics['memory'])
                
                # Visualize results
                print(f"\nVisualizing distributions for {method_name}...")
                self.visualize_cluster_distribution(labels, f"Class Distribution in Clusters - {method_name}")
                self.visualize_2d_clusters(feature_data, labels, f"2D Cluster Visualization - {method_name}")
                self.visualize_2d_clusters(
                    feature_data, 
                    self.true_labels, 
                    f"2D Visualization with True Labels - {method_name}", 
                    use_true_labels=True
                )

            elif method == 'PCA-KMeans':
                # PCA + K-Means on original data
                method_name = "PCA + K-means"
                print(f"\nRunning {method_name}...")
                
                # Apply PCA
                reduced_data = self.apply_pca(self.original_data, n_components=pca_components)
                
                # Run K-Means
                labels, resource_metrics = self.run_kmeans(reduced_data)
                eval_metrics = self.evaluate_clustering(
                    reduced_data, self.true_labels, labels, method_name
                )
                self.export_cluster_distribution_to_csv(labels, f"K-means on {method_name}")
                combined_metrics = {**resource_metrics, **eval_metrics}
                results[method_name] = {
                    'labels': labels,
                    'metrics': combined_metrics
                }
                
                # Add to comparison table
                comparison_metrics['Method'].append(method_name)
                comparison_metrics['ARI'].append(eval_metrics['ARI'])
                comparison_metrics['NMI'].append(eval_metrics['NMI'])
                comparison_metrics['Silhouette'].append(eval_metrics['Silhouette'])
                comparison_metrics['CHI'].append(eval_metrics['CHI'])
                comparison_metrics['DBI'].append(eval_metrics['DBI'])
                comparison_metrics['Time (s)'].append(resource_metrics['time'])
                comparison_metrics['Memory (MB)'].append(resource_metrics['memory'])

                
                # Visualize results
                print(f"\nVisualizing distributions for {method_name}...")
                self.visualize_cluster_distribution(labels, f"Class Distribution in Clusters - {method_name}")
                self.visualize_2d_clusters(reduced_data, labels, f"2D Cluster Visualization - {method_name}")
                self.visualize_2d_clusters(
                    reduced_data, 
                    self.true_labels, 
                    f"2D Visualization with True Labels - {method_name}", 
                    use_true_labels=True
                )

            elif method == 'PCA-Spectral':
                # PCA + Spectral Clustering on original data
                method_name = "PCA + Spectral Clustering"
                print(f"\nRunning {method_name}...")
                
                # Apply PCA
                reduced_data = self.apply_pca(self.original_data, n_components=pca_components)
                
                # Run Spectral Clustering
                labels, resource_metrics = self.run_spectral_clustering(reduced_data)
                eval_metrics = self.evaluate_clustering(
                    reduced_data, self.true_labels, labels, method_name
                )
                self.export_cluster_distribution_to_csv(labels, f"Spectral Clustering on {method_name}")
                combined_metrics = {**resource_metrics, **eval_metrics}
                results[method_name] = {
                    'labels': labels,
                    'metrics': combined_metrics
                }
                
                # Add to comparison table
                comparison_metrics['Method'].append(method_name)
                comparison_metrics['ARI'].append(eval_metrics['ARI'])
                comparison_metrics['NMI'].append(eval_metrics['NMI'])
                comparison_metrics['Silhouette'].append(eval_metrics['Silhouette'])
                comparison_metrics['CHI'].append(eval_metrics['CHI'])
                comparison_metrics['DBI'].append(eval_metrics['DBI'])
                comparison_metrics['Time (s)'].append(resource_metrics['time'])
                comparison_metrics['Memory (MB)'].append(resource_metrics['memory'])
                
                # Visualize results
                print(f"\nVisualizing distributions for {method_name}...")
                self.visualize_cluster_distribution(labels, f"Class Distribution in Clusters - {method_name}")
                self.visualize_2d_clusters(reduced_data, labels, f"2D Cluster Visualization - {method_name}")
                self.visualize_2d_clusters(
                    reduced_data, 
                    self.true_labels, 
                    f"2D Visualization with True Labels - {method_name}", 
                    use_true_labels=True
                )

        # Convert comparison metrics to DataFrame and return
        comparison_df = pd.DataFrame(comparison_metrics)
        print("\nComparison of all methods:")
        print(comparison_df)
        
        # Save comparison to CSV
        comparison_df.to_csv(os.path.join(self.output_dir, "clustering_comparison.csv"), index=False)
        
        return results, comparison_df

# Usage example
if __name__ == "__main__":
    experiment = EnhancedClusteringExperiment(
        test_data_path='data/fashion-mnist_test.csv',
        features_paths={
            'MoCo Pretraining Embeddings': 'embeddings/moco_pretraining_embeddings.csv',
            'MoCo K-means Finetuning Embeddings': 'embeddings/kmeans_finetuning_embeddings.csv',
            'MoCo Spectral Finetuning Embeddings': 'embeddings/spectral_finetuning_embeddings.csv'
        },
        output_dir='exp1_3_results'
    )
    
    # Run full experiment with all methods
    # results, comparison = experiment.run_experiment()
    
    # Or run experiment with selected methods
    results, comparison = experiment.run_experiment(
        methods=['PCA-KMeans', 'PCA-Spectral', 'MoCo Pretraining Embeddings', 'MoCo K-means Finetuning Embeddings', 'MoCo Spectral Finetuning Embeddings'],
        # methods=['MoCo Pretraining Embeddings', 'MoCo K-means Finetuning Embeddings', 'MoCo Spectral Finetuning Embeddings'],#, 'MoCo Pretraining Embeddings', 'MoCo K-means Finetuning Embeddings', 'MoCo Spectral Finetuning Embeddings'],
        pca_components=128
    )