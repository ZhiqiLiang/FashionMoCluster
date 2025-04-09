import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import SpectralClustering
import csv
import os
import argparse

'''
Step1: dataset.py
'''
# Import classes and functions from existing code

class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        
        # Compute the pixel mean for the entire dataset
        images = self.data.iloc[:, 1:].values.reshape(-1, 28, 28).astype('float32')
        self.pixel_mean = images.mean()

        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert to numpy array and subtract the mean
        image = self.data.iloc[idx, 1:].values.reshape(28, 28).astype('float32')
        image = image - self.pixel_mean  # Subtract pixel mean
        
        # Scale the range to [0, 255]
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype('uint8')
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            q, k = self.transform(image)
            return q, k
        return image

# Create a new dataset class for getting embeddings without transformations
class EmbeddingDataset(FashionMNISTDataset):
    def __init__(self, csv_file):
        super().__init__(csv_file, transform=None)
        # Simplified transformation for embedding generation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.pixel_mean], std=[1.0])
        ])
    
    def __getitem__(self, idx):
        # Get the image without heavy augmentations
        image = self.data.iloc[idx, 1:].values.reshape(28, 28).astype('float32')
        image = image - self.pixel_mean
        
        # Normalize to [0, 255]
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype('uint8')
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            image = self.transform(image)
        
        # Return image and label for clustering
        label = self.data.iloc[idx, 0]
        return image, label

'''
Step2: model.py
'''

class ResidualBlock(nn.Module):
    expansion = 1 # Expansion factor of output channels relative to input channels (default: equal channels); in more complex versions of ResNet, self.expansion is set to 4, to expand channels via 1x1 convolutions

    def __init__(self, in_channels, out_channels, stride=1): # The original ResNet paper shows Res18 and Res34 each residual block has two 3x3 convolutions, while Res50, Res101, and Res152 have 1x1, 3x3, and 1x1 convolutions.
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # The first convolution adjusts height and width
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=3, stride=1, padding=1, bias=False) # In Res50, 101, and 152, the last convolution in each residual block often expands the channels by a factor of four
        self.bn2 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels: # For downsampling of x or adjusting the channel size with the same downsampling factor
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False), # 1x1 convolution to match channels
                nn.BatchNorm2d(self.expansion * out_channels) 
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # Add input x
        out = F.relu(out)
        return out

# Improved: Projection Head
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)

class ResNet(nn.Module):
    def __init__(self, residual_block, num_blocks, projection=True):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.projection = projection

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(residual_block, 16, num_blocks[0], stride=1) # Stage 1 has num_blocks[0] residual blocks, with stride=1, i.e., no change in image width/height
        self.layer2 = self._make_layer(residual_block, 32, num_blocks[1], stride=2) # Stage 2 has num_blocks[1] residual blocks, with stride=2, reducing width/height for the first block
        self.layer3 = self._make_layer(residual_block, 64, num_blocks[2], stride=2)
        
        self.linear = nn.Linear(64 * 7 * 7, 128)
        if projection:
            self.projector = ProjectionHead()

    def _make_layer(self, residual_block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1) # The first residual block in each stage has stride=2 to halve width/height, others have stride=1
        layers = []
        for stride in strides:
            layers.append(residual_block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * residual_block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_before_head=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.normalize(out, dim=-1)
        
        if return_before_head:
            return out
            
        if hasattr(self, 'projector') and self.projection:
            out = self.projector(out)
        
        return out

def ResNet32(projection=True): # Set projection=False during inference or downstream tasks
    return ResNet(ResidualBlock, [5, 5, 5], projection) # Options: 3, 5, 7, 9 residual blocks in each stage

# Function to generate embeddings for the entire dataset
def generate_embeddings(encoder, dataloader, device, keep_grad=True):
    # Store previous training state
    previous_state = encoder.training
    
    # Set to eval mode but keep track of gradients if needed
    encoder.eval()
    
    features_list = []
    labels_list = []
    
    # Use appropriate context manager based on keep_grad
    with torch.enable_grad() if keep_grad else torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Get encoder features (without projection head)
            features = encoder(images, return_before_head=True)
            
            # Important: Don't detach features if we need gradients
            features_list.append(features)
            labels_list.append(labels)
    
    # Restore previous training state
    if previous_state:
        encoder.train()
    
    # Concatenate features from all batches
    all_features = torch.cat(features_list, dim=0)
    
    # Ensure labels are on the correct device
    all_labels = torch.cat(labels_list, dim=0).to(device)
    
    return all_features, all_labels

# Custom implementation of clustering metrics in PyTorch
def calculate_cluster_metrics(features, cluster_labels, cluster_centers):
    device = features.device
    n_samples = features.size(0)
    n_clusters = len(torch.unique(cluster_labels))
    
    # Convert cluster_labels to tensor if it's not already
    if not isinstance(cluster_labels, torch.Tensor):
        cluster_labels = torch.tensor(cluster_labels, device=device)
    
    # Convert cluster_centers to tensor if it's not already
    if not isinstance(cluster_centers, torch.Tensor):
        cluster_centers = torch.tensor(cluster_centers, device=device)
    
    # Calculate distances from each point to its assigned cluster center
    distances_to_center = torch.zeros(n_samples, device=device)
    for i in range(n_clusters):
        cluster_mask = (cluster_labels == i)
        if cluster_mask.sum() > 0:
            cluster_points = features[cluster_mask]
            center = cluster_centers[i].unsqueeze(0)
            # Calculate squared Euclidean distance
            dist = torch.sum((cluster_points - center) ** 2, dim=1)
            distances_to_center[cluster_mask] = torch.sqrt(dist)
    
    # Silhouette Score calculation
    a = torch.zeros(n_samples, device=device)  # Mean distance to points in the same cluster
    b = torch.full((n_samples,), float('inf'), device=device)  # Mean distance to points in the nearest cluster
    
    for i in range(n_clusters):
        cluster_mask = (cluster_labels == i)
        cluster_size = cluster_mask.sum().item()
        
        if cluster_size > 1:
            cluster_points = features[cluster_mask]
            
            # Calculate mean intra-cluster distance (a)
            pdist = torch.cdist(cluster_points, cluster_points)
            # Exclude self-distance
            mask = ~torch.eye(cluster_size, dtype=bool, device=device)
            a[cluster_mask] = torch.sum(pdist * mask.float(), dim=1) / (cluster_size - 1)
        
        # Calculate mean distance to points in other clusters (b)
        for j in range(n_clusters):
            if j != i:
                other_cluster_mask = (cluster_labels == j)
                other_cluster_size = other_cluster_mask.sum().item()
                
                if other_cluster_size > 0:
                    other_cluster_points = features[other_cluster_mask]
                    distances_between_clusters = torch.cdist(cluster_points, other_cluster_points)
                    mean_dist_to_cluster_j = torch.mean(distances_between_clusters, dim=1)
                    b[cluster_mask] = torch.minimum(b[cluster_mask], mean_dist_to_cluster_j)
    
    # Calculate silhouette coefficient for each sample
    s = torch.zeros(n_samples, device=device)
    valid_mask = (torch.max(a, b) > 0)
    s[valid_mask] = (b[valid_mask] - a[valid_mask]) / torch.max(a[valid_mask], b[valid_mask])
    
    # Mean silhouette score
    silhouette = torch.mean(s)
    
    # Davies-Bouldin Index calculation
    db_index = 0.0
    for i in range(n_clusters):
        max_ratio = 0.0
        cluster_i_mask = (cluster_labels == i)
        if cluster_i_mask.sum() == 0:
            continue
            
        # Calculate cluster dispersion (average distance from points to centroid)
        cluster_i_dispersion = torch.mean(distances_to_center[cluster_i_mask])
        
        for j in range(n_clusters):
            if i != j:
                cluster_j_mask = (cluster_labels == j)
                if cluster_j_mask.sum() == 0:
                    continue
                
                # Calculate cluster j dispersion
                cluster_j_dispersion = torch.mean(distances_to_center[cluster_j_mask])
                
                # Calculate distance between centroids
                centroid_distance = torch.norm(cluster_centers[i] - cluster_centers[j])
                
                if centroid_distance > 0:
                    # Calculate ratio of cluster dispersions to centroid distance
                    ratio = (cluster_i_dispersion + cluster_j_dispersion) / centroid_distance
                    max_ratio = max(max_ratio, ratio)
        
        db_index += max_ratio
    
    # Normalize by number of clusters
    if n_clusters > 1:
        db_index /= n_clusters
    
    # Calinski-Harabasz Index calculation
    # Between-cluster dispersion
    cluster_variances = torch.zeros(n_clusters, device=device)
    cluster_sizes = torch.zeros(n_clusters, device=device)
    
    global_mean = torch.mean(features, dim=0)
    
    # Calculate between-cluster variance
    between_cluster_var = torch.tensor(0.0, device=device)
    for i in range(n_clusters):
        cluster_mask = (cluster_labels == i)
        cluster_size = cluster_mask.sum().item()
        
        if cluster_size > 0:
            cluster_sizes[i] = cluster_size
            center_diff = cluster_centers[i] - global_mean
            between_cluster_var += cluster_size * torch.sum(center_diff ** 2)
    
    # Calculate within-cluster variance
    within_cluster_var = torch.sum(distances_to_center ** 2)
    
    # Calculate CH score
    if within_cluster_var > 0 and n_samples > n_clusters:
        ch_score = (between_cluster_var / (n_clusters - 1)) / (within_cluster_var / (n_samples - n_clusters))
    else:
        ch_score = torch.tensor(0.0, device=device)
    
    # Normalize CH score
    ch_norm = torch.log1p(ch_score) / 10.0
    
    # Invert DB index for loss calculation (lower is better, so invert it)
    db_loss = 1.0 / (db_index + 1e-10)
    
    return silhouette, db_loss, ch_norm

# Clustering loss function
def calculate_clustering_loss(features, labels, n_clusters=10):
    device = features.device
    
    # If features has gradients, make a copy for numpy operations
    features_for_clustering = features.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Perform Spectral clustering
    spectral_clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=10,
        random_state=42
    )
    cluster_labels = spectral_clustering.fit_predict(features_for_clustering)
    
    # Since Spectral Clustering doesn't produce cluster centers directly, we need to compute them
    centers = []
    for i in range(n_clusters):
        mask = (cluster_labels == i)
        if np.sum(mask) > 0:
            center = np.mean(features_for_clustering[mask], axis=0)
            centers.append(center)
        else:
            # If no points in cluster, add a dummy center
            centers.append(np.zeros_like(features_for_clustering[0]))
    
    cluster_centers = np.array(centers)
    
    # Convert to PyTorch tensors
    cluster_centers = torch.tensor(cluster_centers, device=device, dtype=features.dtype)
    cluster_labels = torch.tensor(cluster_labels, device=device)
    
    # Calculate custom metrics with gradient preservation
    sil, dbi_loss, chi_norm = calculate_cluster_metrics(features, cluster_labels, cluster_centers)
    
    # Calculate NMI and ARI (no gradients needed for these)
    nmi = normalized_mutual_info_score(labels_np, cluster_labels.cpu().numpy())
    ari = adjusted_rand_score(labels_np, cluster_labels.cpu().numpy())
    
    # Use silhouette score as the loss function
    cluster_score = sil
    
    # Convert to loss (we want to maximize the metrics, so minimize negative)
    clustering_loss = -1.0 * cluster_score
    
    metrics = {
        'nmi': nmi,
        'ari': ari,
        'silhouette': sil.item(),
        'davies_bouldin_inv': dbi_loss.item(),
        'calinski_harabasz_norm': chi_norm.item()
    }
    
    return clustering_loss, metrics

'''
Step3: main.py
'''
def main(args):
    # Create GPU environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define paths
    data_path = args.data_path
    model_save_path = Path(args.model_save_path)
    model_save_path.mkdir(parents=True, exist_ok=True)
    pretrained_model_path = model_save_path / 'encoder_q_best.pt'
    embedding_save_path = Path(args.embedding_save_path)
    embedding_save_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataset and dataloader for embeddings (without heavy augmentations)
    embedding_dataset = EmbeddingDataset(data_path)
    embedding_dataloader = torch.utils.data.DataLoader(embedding_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize encoder network
    encoder = ResNet32(projection=False).to(device)  # Important: set projection=False to remove adapter
    
    # Load pre-trained parameters - handling the case where projection head exists in saved model
    if os.path.exists(pretrained_model_path):
        # Load the full state dict from the saved model
        state_dict = torch.load(pretrained_model_path, map_location=device)
        # Filter out the projection head parameters
        encoder_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('projector')}
        # Load the filtered state dict
        missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded pre-trained parameters from {pretrained_model_path}")
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    else:
        print(f"Pre-trained model not found at {pretrained_model_path}. Starting with random weights.")
    
    # Optimizer with smaller learning rate for fine-tuning
    optimizer = optim.SGD(encoder.parameters(), 
                          lr=args.lr,  # Reduced learning rate for fine-tuning
                          momentum=0.9,
                          weight_decay=0.0001)
    
    # Early stopping parameters
    patience = args.patience
    min_delta = 0.0001
    best_metrics = {'nmi': 0, 'ari': 0, 'silhouette': -1, 'davies_bouldin_inv': 0, 'calinski_harabasz_norm': 0}
    counter = 0
    
    # For logging clustering metrics
    clustering_metrics_history = []
    
    # Start fine-tuning with clustering loss
    for epoch in range(args.num_epochs):
        encoder.train()  # Set to training mode
        
        # Generate embeddings while preserving gradients
        features, labels = generate_embeddings(
            encoder, embedding_dataloader, device, keep_grad=True
        )
        
        # Calculate clustering loss with gradient-preserving metrics
        clustering_loss, metrics = calculate_clustering_loss(
            features, labels, n_clusters=args.num_clusters
        )
        
        # Apply clustering loss
        optimizer.zero_grad()
        clustering_loss.backward()
        optimizer.step()
        
        # Add metrics to history
        metrics_entry = {**metrics, 'epoch': epoch}
        clustering_metrics_history.append(metrics_entry)
        
        # Log metrics
        print(f"Epoch {epoch+1}/{args.num_epochs} Clustering Metrics: "
              f"NMI={metrics['nmi']:.4f}, "
              f"ARI={metrics['ari']:.4f}, "
              f"Silhouette={metrics['silhouette']:.4f}, "
              f"Davies-Bouldin Inv={metrics['davies_bouldin_inv']:.4f}, "
              f"Calinski-Harabasz Norm={metrics['calinski_harabasz_norm']:.4f}, "
              f"Loss={clustering_loss.item():.6f}")
        
        # Save model if we have improvement in clustering metrics
        combined_metric = (0.3 * metrics['nmi'] + 0.3 * metrics['ari'] + 0.3 * metrics['silhouette'] +
                           0.05 * metrics['davies_bouldin_inv'] + 0.05 * metrics['calinski_harabasz_norm'])
        best_combined_metric = (0.3 * best_metrics['nmi'] + 0.3 * best_metrics['ari'] + 0.3 * best_metrics['silhouette'] + 
                                0.05 * best_metrics['davies_bouldin_inv'] + 0.05 * best_metrics['calinski_harabasz_norm'])
        
        if combined_metric > (best_combined_metric + min_delta):
            best_metrics = metrics.copy()
            counter = 0
            torch.save(encoder.state_dict(), model_save_path / 'encoder_spectral_best.pt')
            print(f"Saved best model at epoch {epoch+1}")
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Save checkpoint every `save_every_n_epochs` epochs
        if (epoch + 1) % args.save_every_n_epochs == 0:
            torch.save(encoder.state_dict(), model_save_path / f'encoder_spectral_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save(encoder.state_dict(), model_save_path / 'encoder_spectral_final.pt')
    
    # Save clustering metrics history
    if clustering_metrics_history:
        metrics_df = pd.DataFrame(clustering_metrics_history)
        metrics_df.to_csv(model_save_path / 'spectral_clustering_metrics.csv', index=False)
    
    # Generate final embeddings for the entire dataset using the best model
    print("Generating final embeddings using the best model...")
    encoder.load_state_dict(torch.load(model_save_path / 'encoder_spectral_best.pt', map_location=device))
    encoder.eval()
    
    with torch.no_grad():
        final_features, final_labels = generate_embeddings(
            encoder, embedding_dataloader, device, keep_grad=False
        )
    
    # Convert to numpy for saving
    features_np = final_features.cpu().numpy()
    labels_np = final_labels.cpu().numpy()
    
    # Save embeddings to CSV
    embeddings_csv_path = embedding_save_path / 'fashion_mnist_spectral_embeddings.csv'
    with open(embeddings_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'feature_{i}' for i in range(features_np.shape[1])]
        writer.writerow(header)
        
        for i in range(len(labels_np)):
            row = [int(labels_np[i])] + features_np[i].tolist()
            writer.writerow(row)
    
    print(f"Embeddings saved to {embeddings_csv_path}")
    
    # Perform final Spectral clustering and evaluation
    print("Performing final Spectral clustering evaluation...")
    spectral_clustering = SpectralClustering(
        n_clusters=args.num_clusters,
        affinity='nearest_neighbors',
        n_neighbors=10,
        random_state=42
    )
    cluster_labels = spectral_clustering.fit_predict(features_np)
    
    # Calculate final metrics
    final_nmi = normalized_mutual_info_score(labels_np, cluster_labels)
    final_ari = adjusted_rand_score(labels_np, cluster_labels)
    final_silhouette = silhouette_score(features_np, cluster_labels)
    final_db = davies_bouldin_score(features_np, cluster_labels)
    final_ch = calinski_harabasz_score(features_np, cluster_labels)
    
    final_db_inv = 1.0 / (final_db + 1e-10)
    final_ch_norm = np.log1p(final_ch) / 10.0
    
    print(f"Final Spectral clustering evaluation:")
    print(f"NMI: {final_nmi:.4f}")
    print(f"ARI: {final_ari:.4f}")
    print(f"Silhouette Score: {final_silhouette:.4f}")
    print(f"Davies-Bouldin Inverse: {final_db_inv:.4f}")
    print(f"Davies-Bouldin: {final_db:.4f}")
    print(f"Calinski-Harabasz Norm: {final_ch_norm:.4f}")
    
    # Save final metrics
    final_metrics = {
        'nmi': final_nmi,
        'ari': final_ari,
        'silhouette': final_silhouette,
        'davies_bouldin_inv': final_db_inv,
        'davies_bouldin': final_db,
        'calinski_harabasz': final_ch,
        'calinski_harabasz_norm': final_ch_norm
    }
    
    pd.DataFrame([final_metrics]).to_csv(embedding_save_path / 'spectral_final_metrics.csv', index=False)
    
    # Save cluster assignments
    cluster_df = pd.DataFrame({
        'label': labels_np,
        'cluster': cluster_labels
    })
    cluster_df.to_csv(embedding_save_path / 'spectral_cluster_assignments.csv', index=False)
    
    print("Spectral clustering fine-tuning completed successfully!")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Fine-tune clustering model and save results")
    parser.add_argument('--data_path', type=str, default='data/fashion-mnist_test.csv', help="Path to the test data")
    parser.add_argument('--model_save_path', type=str, default="model_weights", help="Path to save model weights")
    parser.add_argument('--embedding_save_path', type=str, default="embeddings", help="Path to save embeddings")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for fine-tuning")
    parser.add_argument('--num_clusters', type=int, default=10, help="Number of clusters for spectral clustering")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for data loader")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--patience', type=int, default=30, help="Patience for early stopping")
    parser.add_argument('--save_every_n_epochs', type=int, default=10, help="Save model every n epochs")
    
    args = parser.parse_args()
    main(args)