# Exploring the Boundaries of Unsupervised Learning for Fashion-MNIST via Momentum Constrast and Cluster-based Finetuning

This repository provides the code implementation of the graduation thesis "Exploring the Boundaries of Unsupervised Learning for Fashion-MNIST via Momentum Constrast and Cluster-based Finetuning", including all the training script (MoCo pretraining, Spectral-Clustering-based finetuning), experiment script (comparison, ablation study, visualization) and necessary data and model weights.

## Installation

To run the code, make sure you have Python 3.8+ and the following dependencies installed. You can set up a virtual environment and install dependencies using the following commands:

```bash
# Clone the GitHub repository to your local machine
git clone https://github.com/ZhiqiLiang/FashionMoCluster.git
cd FashionMoCluster

# Create and activate a virtual environment
python3 -m venv fashion_MoCluster_env
source fashion_MoCluster_env/bin/activate  # On Windows, use `fashion_mnist_env\Scripts\activate`

# Install necessary dependencies
pip install -r requirements.txt
```

## Directory Structure

The project directory structure is as follows:

```graghql
FashionMoCluster/
│
├── data/                     # Directory for dataset (e.g., Fashion MNIST)
│   ├── data.csv  # script to download Fashion-MNIST dataset
│   ├── fashion-mnist_test.csv  # Fashion MNIST test dataset (CSV format)
│
├── embeddings/                # Directory for storing embeddings
│   ├── moco_pretraining_embeddings.csv
│   ├── kmeans_finetuning_embeddings.csv
│   ├── spectral_finetuning_embeddings.csv
│
├── model_weights/            # Directory for saving trained models
│   ├── encoder_q_best.pt # Best model for MoCo pretraining
│   ├── encoder_clustering_best.pt # Best model for K-Means finetuning
│   ├── encoder_spectral_best.pt # Best model for Spectral finetuning
│
├── src/                      # Source code files
│   ├── train_pretraining_moco.py # train script for moco pretraining
│   ├── train_finetuning_spectral_clusteting.py # train script for spectral finetuning
│   ├── exp_1_3.py # experiment 1 and 3 script
│   ├── exp_2.py # experiment 2 script
│
├── README.md                 # This README file
└── requirements.txt          # Python dependencies for the project
```

## Space and memory requirements

All the training and testing are done on a single node with 1 NVIDIA RTX 4090 GPUs (25.2GB RAM/GPU) and 1 20-core 2.5GHz Intel Xeon CPUs (768GB RAM).

## Code Manual

### 1. File Overview: `train_pretraining_moco.py`

#### (1) Code Summary:

This Python script is designed to train a model using MoCo (Momentum Contrast) with grayscale augmentation on the FashionMNIST dataset. The model is a deep ResNet architecture, and the training follows a contrastive learning setup where pairs of augmented images are processed to learn useful representations. The script leverages a custom augmentation pipeline, `GrayscaleSimCLRAugmentation`, to apply transformations like Gaussian blur, noise addition, and Sobel filtering for both query and key images. The script incorporates techniques like momentum-based update for the key encoder and temperature annealing for contrastive loss calculation. Additionally, early stopping and checkpoint saving strategies are implemented for model optimization.

#### (2) Running Instructions and Crucial Constants:

To run the script, use the following command:

```bash
python train_pretraining_moco.py --dataset_path <path_to_csv_file> --batch_size <batch_size> --num_epochs <epochs> --lr <learning_rate> --patience <patience_epochs> --save_every_n_epochs <checkpoint_frequency> --model_save_path <model_save_directory>
```

**Crucial Constants (and their location in the code):**

- `dataset_path`: Path to the FashionMNIST dataset CSV file (e.g., `data/fashion-mnist_test.csv`). Defined in `args.dataset_path`.
- `batch_size`: The batch size used during training. Defined in `args.batch_size`.
- `num_epochs`: The total number of epochs for training. Defined in `args.num_epochs`.
- `lr`: The initial learning rate. Defined in `args.lr`.
- `patience`: The number of epochs with no improvement after which training will stop early. Defined in `args.patience`.
- `save_every_n_epochs`: Frequency of saving model checkpoints. Defined in `args.save_every_n_epochs`.
- `model_save_path`: Path to save model weights during training (e.g., `model_weights`). Defined in `args.model_save_path`.
- `queue_size`: Size of the queue used in contrastive learning. Set to 8192 in the code.
- `m`: Momentum coefficient for updating the key encoder. Set to 0.999 in the code.

#### (3) Main Classes, Procedures, Methods, and Data Structures:

- **Classes:**
  - `FashionMNISTDataset`: A custom dataset class that loads and preprocesses FashionMNIST data from a CSV file. It normalizes images by subtracting the dataset's pixel mean and scales pixel values to the [0, 255] range. The `__getitem__` method returns augmented query (`q`) and key (`k`) images for contrastive learning.
  - `ResidualBlock`: Implements a residual block used in the ResNet architecture. It contains two convolutional layers with batch normalization and an optional shortcut connection.
  - `ProjectionHead`: A small multi-layer perceptron (MLP) used to project the output features from the ResNet into a normalized vector space, commonly used in contrastive learning.
  - `ResNet`: A ResNet-based model that consists of several residual blocks (layers) and a final linear layer. The model can be configured with or without a projection head (using `ProjectionHead`).
  - `GrayscaleSimCLRAugmentation`: A custom augmentation pipeline designed for contrastive learning with grayscale images. It applies transformations like random resized cropping, random rotation, horizontal flipping, Gaussian blur, noise addition, and Sobel filtering.
- **Procedures/Methods:**
  - `temperature_schedule`: A function that defines the temperature annealing schedule for the contrastive loss.
  - `contrastive_loss`: Computes the contrastive loss given query (`q`), key (`k`), and a queue of previous key embeddings. It calculates the positive and negative similarities and returns the cross-entropy loss.
  - `main`: The main training procedure that:
    - Sets up the GPU environment.
    - Initializes dataset, dataloaders, and model (query and key encoders).
    - Defines the optimizer (SGD) and learning rate scheduler (Cosine Annealing).
    - Implements early stopping and model checkpoint saving based on loss.
    - Performs the forward pass, loss computation, backward pass, and updates for both query and key encoders.
  - `ResNet32`: A function that creates a ResNet model with 32 layers.
- **Data Structures:**
  - `dataloader`: The DataLoader object used to iterate through the dataset during training, providing batches of images.
  - `queue`: A tensor used to store and normalize the embeddings of key images from previous iterations for contrastive learning.
  - `queue_ptr`: A pointer used to track the current position in the queue where new key embeddings are inserted.
- **Key Flow:**
  - The dataset is loaded from the CSV file, and data augmentations are applied to generate pairs of query and key images.
  - The model is trained using a contrastive learning approach where query and key embeddings are learned through a contrastive loss function. The model's weights are updated through backpropagation using the SGD optimizer.
  - Key encoder weights are updated using a momentum-based update mechanism.
  - Early stopping is implemented to halt training when the loss does not improve for a certain number of epochs.
  - The model is saved at regular intervals and the best model is stored based on loss improvement.

### 2. File Overview: `train_finetuning_spectral_clustering.py`

#### (1) Code Summary:

This script performs fine-tuning for a pre-trained model using spectral clustering as a loss function. The goal is to fine-tune an encoder network (a ResNet-based architecture) using the embeddings of FashionMNIST images and the clustering loss metrics based on spectral clustering. The fine-tuning process improves the quality of the embeddings by optimizing the clustering performance over multiple epochs. The resulting embeddings and metrics are saved for further analysis.

#### (2) Running Instructions and Crucial Constants:

To run the script, use the following command:

```bash
python train_finetuning_spectral_clustering.py --data_path <path_to_data> --model_save_path <path_to_save_model> --embedding_save_path <path_to_save_embeddings> --batch_size <batch_size> --lr <learning_rate> --num_epochs <epochs> --num_clusters <num_clusters> --save_every_n_epochs <save_interval> --patience <early_stopping_patience>
```

Here are the crucial constants and their locations:

* `data_path`: Path to the dataset for generating embeddings (passed as a command-line argument).
* `model_save_path`: Directory where the fine-tuned model and checkpoints are saved (passed as a command-line argument).
* `embedding_save_path`: Directory to save the generated embeddings and final clustering results (passed as a command-line argument).
* `batch_size`: The batch size used for generating embeddings (passed as a command-line argument).
* `lr`: Learning rate for fine-tuning (passed as a command-line argument).
* `num_epochs`: The number of epochs to fine-tune the model (passed as a command-line argument).
* `num_clusters`: The number of clusters for spectral clustering (passed as a command-line argument).
* `save_every_n_epochs`: How frequently to save model checkpoints (passed as a command-line argument).
* `patience`: Early stopping patience to stop training if no improvement is observed (passed as a command-line argument).

#### (3) Main Classes, Procedures, Methods, and Data Structures:

* **Classes:**
  - `EmbeddingDataset`: A custom dataset class that loads the FashionMNIST dataset and applies minimal transformations for generating embeddings. The `__getitem__` method retrieves a transformed image and its label for use in embedding generation.

* **Procedures/Methods:**

  - `generate_embeddings`: A function that generates embeddings for the entire dataset using the encoder model. It takes the encoder, data loader, and device, and outputs embeddings and corresponding labels.

  - `calculate_cluster_metrics`: A function that computes various clustering evaluation metrics (Silhouette score, Davies-Bouldin index, and Calinski-Harabasz index) for the clustering performance. The metrics calculated are:
    - **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters.
    - **Davies-Bouldin Index**: A metric to evaluate cluster separation, where a lower score indicates better clustering.
    - **Calinski-Harabasz Score**: Evaluates the dispersion of clusters, where higher values indicate better clustering.

  - `calculate_clustering_loss`: A function that calculates the loss function for clustering (Spectral Clustering) based on the generated embeddings. It takes features and labels from the encoder, and the number of clusters, and outputs the clustering loss and a dictionary containing clustering metrics (e.g., NMI, ARI, Silhouette score).

  - `main`: The entry point of the script. It manages the entire fine-tuning process:
    - Loads the dataset and dataloaders.
    - Initializes the encoder model and loads pre-trained weights.
    - Fine-tunes the model using the spectral clustering loss function.
    - Saves model checkpoints and final embeddings.
    - Evaluates the final clustering performance.

* **Data Structures:**

  - `dataloader`: The DataLoader object used to iterate through the dataset during training, providing batches of images.

  - `embeddings`: A tensor that stores the generated embeddings for each image in the dataset.

  - `metrics`: A dictionary storing clustering evaluation metrics such as Silhouette score, Davies-Bouldin index, and Calinski-Harabasz score.

  - `model`: The encoder model used to generate embeddings and fine-tune the clustering task.

* **Key Flow:**
  * Loads pre-trained model weights.
  * Generates embeddings for FashionMNIST dataset using the encoder network.
  * Applies spectral clustering to the generated embeddings.
  * Fine-tunes the encoder by optimizing a clustering loss based on metrics such as Silhouette score, Davies-Bouldin index, and Calinski-Harabasz index.
  * Saves the fine-tuned model, embeddings, and clustering metrics at various points during training.

### 3. File Overview: `exp_1_3.py`

#### (1) Code Summary:

This code implements a comprehensive clustering experiment for evaluating various unsupervised learning techniques on fashion visual data, specifically focusing on MoCo pretraining and finetuning strategies, combined with K-means and Spectral clustering methods. The primary goal is to assess the clustering performance of these methods on the dataset and visualize the results in a meaningful way. It includes methods for data preprocessing, applying PCA, running clustering algorithms, evaluating clustering performance using different metrics, and visualizing the clustering results.

#### (2) Running Instructions and Crucial Constants:

To run the script, use the following:

```python
if __name__ == "__main__":
    experiment = EnhancedClusteringExperiment(
        test_data_path='path_to_test_data.csv',
        features_paths={
            'MoCo Pretraining Embeddings': 'path_to_embeddings.csv'
        },
        output_dir='output_directory'
    )
    results, comparison = experiment.run_experiment()
    
# Example usage:
experiment = EnhancedClusteringExperiment(
    test_data_path='data/fashion-mnist_test.csv',
    features_paths={
        'MoCo Pretraining Embeddings': 'embeddings/moco_pretraining_embeddings.csv',
        'MoCo K-means Finetuning Embeddings': 'embeddings/kmeans_finetuning_embeddings.csv',
        'MoCo Spectral Finetuning Embeddings': 'embeddings/spectral_finetuning_embeddings.csv'
    },
    output_dir='exp_1_3_results'
)

# Run experiment with selected methods
results, comparison = experiment.run_experiment(
    methods=['PCA-KMeans', 'PCA-Spectral', 'MoCo Pretraining Embeddings']
)
```

Here are the key constants:

- `n_clusters`: Specifies the number of clusters (default: 10).
- `pca_components`: Number of components to keep when performing PCA (default: 128).
- `methods`: A list of the clustering methods to be used, including options like 'PCA-KMeans', 'PCA-Spectral', and 'MoCo Pretraining Embeddings'.

#### (3) Main Classes, Procedures, Methods, and Data Structures:

* **Main Classes**
  - **EnhancedClusteringExperiment**: The primary class that encapsulates the entire clustering experiment.
    - **Attributes**:
      - `test_data_path`: Path to the test dataset CSV.
      - `features_paths`: Dictionary mapping feature names to file paths for pre-trained, finetuned, or PCA features.
      - `output_dir`: Directory where output and visualizations will be saved.
    - **Methods**:
      - `load_data()`: Loads the test data and features from CSV files.
      - `preprocess_data()`: Scales the data and feature arrays using `StandardScaler`.
      - `apply_pca()`: Applies PCA on the data and reduces its dimensionality.
      - `run_kmeans()`: Runs K-means clustering on the provided data.
      - `run_spectral_clustering()`: Runs Spectral clustering on the provided data.
      - `evaluate_clustering()`: Evaluates clustering performance using metrics like ARI, NMI, Silhouette, etc.
      - `visualize_2d_clusters()`: Creates 2D visualizations of the clustered data.
      - `export_cluster_distribution_to_csv()`: Exports the distribution of true labels in each cluster to a CSV file.
      - `visualize_cluster_distribution()`: Visualizes the distribution of labels across clusters.
      - `run_experiment()`: Runs the clustering experiment for multiple methods, collects results, and compares them.

* **Key Methods**

  - **`load_data()`**: Loads the data, including both the test data and any feature data (such as MoCo embeddings).

  - **`run_kmeans()`**: Runs K-means clustering on the feature data and tracks resource usage.

  - **`run_spectral_clustering()`**: Runs Spectral clustering on the feature data and tracks resource usage.

  - **`evaluate_clustering()`**: Calculates clustering evaluation metrics such as ARI, NMI, and Silhouette score.

  - **`visualize_2d_clusters()`**: Visualizes the clustering results in a 2D plot.

  - **`visualize_cluster_distribution()`**: Visualizes the distribution of true labels within each cluster.

  - **`export_cluster_distribution_to_csv()`**: Saves the label distribution for each cluster into a CSV file.

* **Data Structures**

  - **`df_test`**: DataFrame containing the test data, including image features and labels.

  - **`feature_dataframes`**: Dictionary storing feature DataFrames for different methods.

  - **`feature_arrays`**: Dictionary storing the actual feature arrays for each method.

  - **`true_labels`**: Array containing the true labels of the test dataset.

  - **`label_names`**: Dictionary mapping label IDs to their human-readable names (e.g., T-shirt, Trouser).

  - **`colors`**: List of colors used for plotting different clusters.

* **Key Flow:** please see the detailed experiment design in the paper.

### 3. File Overview: `exp_2.py`

#### (1) Code Summary:

This code is designed to conduct an ablation study on the impact of various architectural components and training strategies in self-supervised learning for contrastive representation learning. The main purpose of the experiments is to explore how different configurations of the model and training process affect performance. Specifically, the study includes the following variations:

(a) **ConvNet only**: The baseline model without any advanced components such as residual connections or projection heads.

(b) **ConvNet + ResConn**: This setup adds residual connections to the network, allowing the model to benefit from deeper architectures.

(c) **ConvNet + ResConn + MLP**: Here, a projection head (MLP) is added to the model to help improve the representation power and aid in contrastive learning.

(d) **ConvNet + ResConn + MLP + Advanced Augmentation**: In this setup, the model uses advanced augmentation strategies like noise, Gaussian blur, and Sobel filters, which help to enhance the robustness of the learned representations.

(e) **ConvNet + ResConn + MLP + Advanced Augmentation + Cosine Annealing**: In addition to the previous setups, this configuration uses cosine annealing for the learning rate schedule, providing dynamic adjustment of the learning rate during training.

(f) **ConvNet + ResConn + MLP + Advanced Augmentation + Cosine Annealing + Larger Model**: The model is scaled up to a larger size, testing the impact of increasing the number of layers and capacity.

(g) **ConvNet + ResConn + MLP + Advanced Augmentation + Cosine Annealing + Larger Model + More Epochs**: This setup extends the training duration to test the effect of longer training on model performance.

The goal of this study is to measure how these factors contribute to the performance on tasks such as clustering and representation learning.

#### (2) Running Instructions and Crucial Constants:

You can train the model by calling the `train_and_evaluate` function, passing the configuration, dataset path, and saving directory as arguments. Here's an example command:

```bash
python train.py --config 'config_a' --data_path '/path/to/data' --save_dir './model_weights'
```

Use the predefined configurations to specify the ablation case. Modify the configuration file or choose from the provided options (`'a'`, `'b'`, `'c'`, etc.) to experiment with different setups.

Here are the crucial constants:

- **Batch Size**: Set via the configuration (default: 256).
- **Learning Rate**: Set via the configuration (default: 0.03).
- **Epochs**: The number of epochs for training (default: 50).
- **Queue Size**: For contrastive learning (default: 8192).
- **Model Size**: Configurable as 20 or 34 layers.

#### (3) Main Classes, Procedures, Methods, and Data Structures:

* **Main Classes**:

  - `BasicAugmentation`: Implements basic augmentation techniques (e.g., random horizontal flip, rotation) for the ablation study configurations (cases a-c).

  - `AdvancedAugmentation`: Applies more complex augmentations such as Gaussian blur, noise, and Sobel filter for cases d and above.

  - `SimpleConvBlock`: A simple convolutional block without residual connections, used in the baseline model (case a).

  - `ResidualBlock`: A convolutional block with skip connections to form residual networks (used in case b and above).

  - `ProjectionHead`: An MLP used for projecting the features into a space suitable for contrastive learning (used in cases c, d, e).

  - `IdentityProjection`: A simple identity projection that bypasses any projection head, used for ablation.

  - `BaseNetwork`: The main model architecture that can be configured with different block types (simple vs. residual) and other components like MLP projection or augmentation strategies.

* **Key Methods**:

  - `get_config(case)`: Returns a configuration dictionary for each ablation study case, where `case` can be any of the predefined configurations ('a' to 'g').

  - `train_and_evaluate(config, data_path, save_dir)`: The core function that trains the model according to the configuration and evaluates the results.

  - `create_model(config)`: Builds the model based on the configuration, selecting between simple or residual blocks and including projection heads or not.

  - `contrastive_loss(q, k, queue, epoch, total_epochs)`: Computes the contrastive loss (InfoNCE) between the query and key pairs, with temperature annealing.

  - `evaluate_clustering(features, labels, n_clusters=10)`: Evaluates the clustering performance of the embeddings produced by the model.

* **Data Structures**:

  - `FashionMNISTDataset`: Custom dataset for loading and transforming the FashionMNIST dataset, which is used in the experiments.

  - `EmbeddingDataset`: Dataset used for generating embeddings from the trained model.

  - `queue`: A memory queue used for contrastive learning to store negative samples, facilitating efficient training.

* **Key Flow**: please see the detailed experiment design in the paper.
