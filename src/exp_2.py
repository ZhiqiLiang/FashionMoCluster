import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
import cv2
from PIL import Image, ImageOps, ImageFilter
import time
import argparse
import torch.cuda.amp as amp
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score

#############################################
# Configuration for Ablation Study
#############################################

def get_config(case):
    """
    Returns configuration for each ablation study case
    
    case (a): ConvNet only
    case (b): ConvNet + ResConn
    case (c): ConvNet + ResConn + MLP
    case (d): ConvNet + ResConn + MLP + aug+
    case (e): ConvNet + ResConn + MLP + aug+ + cos
    case (f): ConvNet + ResConn + MLP + aug+ + cos + larger size
    case (g): ConvNet + ResConn + MLP + aug+ + cos + larger size + more epochs
    """
    # Default configuration
    config = {
        'use_convnet': True,
        'use_resconn': False,
        'use_mlp': False,
        'use_advanced_aug': False,
        'use_cosine_lr': False,
        'model_size': 20,  # Number of layers
        'epochs': 50,
        'queue_size': 8192,
        'batch_size': 256, 
        'lr': 0.03,
    }
    
    if case == 'a':
        # ConvNet only - base setup
        pass
    elif case == 'b':
        # Add residual connections
        config['use_resconn'] = True
    elif case == 'c':
        # Add projection head MLP
        config['use_resconn'] = True
        config['use_mlp'] = True
    elif case == 'd':
        # Add advanced augmentation
        config['use_resconn'] = True
        config['use_mlp'] = True
        config['use_advanced_aug'] = True
    elif case == 'e':
        # Add cosine annealing
        config['use_resconn'] = True
        config['use_mlp'] = True
        config['use_advanced_aug'] = True
        config['use_cosine_lr'] = True
    elif case == 'f':
        # Larger model
        config['use_resconn'] = True
        config['use_mlp'] = True
        config['use_advanced_aug'] = True
        config['use_cosine_lr'] = True
        config['model_size'] = 34
    elif case == 'g':
        # More training epochs
        config['use_resconn'] = True
        config['use_mlp'] = True
        config['use_advanced_aug'] = True
        config['use_cosine_lr'] = True
        config['model_size'] = 34
        config['epochs'] = 300
    
    return config

#############################################
# Dataset Implementation
#############################################

class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        
        # Calculate mean pixel value across the dataset
        images = self.data.iloc[:, 1:].values.reshape(-1, 28, 28).astype('float32')
        self.pixel_mean = images.mean()
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert to numpy array and subtract mean
        image = self.data.iloc[idx, 1:].values.reshape(28, 28).astype('float32')
        image = image - self.pixel_mean
        
        # Normalize to [0, 255]
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype('uint8')
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            q, k = self.transform(image)
            return q, k
        return image

# Dataset for generating embeddings without transformations
class EmbeddingDataset(FashionMNISTDataset):
    def __init__(self, csv_file):
        super().__init__(csv_file, transform=None)
        # Simplified transformation for embedding generation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.pixel_mean/255.0], std=[1.0])
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

#############################################
# Data Augmentation
#############################################

class BasicAugmentation:
    """Basic augmentation for case (a), (b), (c)"""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    
    def __call__(self, x):
        q = self.transform(x)
        k = self.transform(x)
        return q, k

class AdvancedAugmentation:
    """Advanced augmentation for cases (d) and above"""
    def __init__(self):
        self.q_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=28, scale=(0.2, 1.0)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(self._gaussian_blur),
            transforms.Lambda(self._add_noise),
            transforms.ToTensor(),
        ])
        
        self.k_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=28, scale=(0.2, 1.0)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(self._sobel_filter),
            transforms.ToTensor(),
        ])

    def _gaussian_blur(self, img):
        img_array = np.array(img)
        blurred = cv2.GaussianBlur(img_array, (5, 5), np.random.uniform(0.1, 2.0))
        return Image.fromarray(blurred)

    def _add_noise(self, img):
        img_array = np.array(img)
        noise = np.random.normal(0, 15, img_array.shape)
        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def _sobel_filter(self, img):
        img_array = np.array(img)
        sobel_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
        
        grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        grad_magnitude = np.clip(grad_magnitude, 0, 255).astype(np.uint8)
        return Image.fromarray(grad_magnitude)

    def __call__(self, x):
        q = self.q_transform(x)
        k = self.k_transform(x)
        return q, k

#############################################
# Model Architecture
#############################################

class SimpleConvBlock(nn.Module):
    """Simple convolutional block without residual connections"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(SimpleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels) 
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # Residual Connection
        out = F.relu(out)
        return out

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning"""
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)

class IdentityProjection(nn.Module):
    """Identity projection (no MLP) for ablation"""
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128):
        super().__init__()
    
    def forward(self, x):
        return x  # Just return the input (already normalized)

class BaseNetwork(nn.Module):
    """Base network that can be configured for different ablation cases"""
    def __init__(self, block_type, num_blocks, use_projection=True, use_mlp=True):
        super(BaseNetwork, self).__init__()
        self.in_channels = 16
        self.use_projection = use_projection

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block_type, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_type, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_type, 64, num_blocks[2], stride=2)
        
        self.linear = nn.Linear(64 * 7 * 7, 128)
        
        if use_projection:
            if use_mlp:
                self.projector = ProjectionHead()
            else:
                self.projector = IdentityProjection()

    def _make_layer(self, block_type, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block_type(self.in_channels, out_channels, stride))
            if block_type == ResidualBlock:
                self.in_channels = out_channels * block_type.expansion
            else:
                self.in_channels = out_channels
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
            
        if hasattr(self, 'projector') and self.use_projection:
            out = self.projector(out)
        
        return out

def create_model(config):
    """Create model based on configuration"""
    # Determine block type
    block_type = ResidualBlock if config['use_resconn'] else SimpleConvBlock
    
    # Determine number of blocks based on model size
    if config['model_size'] == 20:
        num_blocks = [3, 3, 3]
    elif config['model_size'] == 34:
        num_blocks = [5, 5, 5]
    else:
        raise ValueError(f"Unsupported model size: {config['model_size']}")
    
    return BaseNetwork(
        block_type=block_type,
        num_blocks=num_blocks,
        use_projection=True,
        use_mlp=config['use_mlp']
    )

#############################################
# Loss Function
#############################################

def temperature_schedule(epoch, total_epochs, initial_temp=0.07, final_temp=0.01):
    """Annealing temperature schedule"""
    return initial_temp * (final_temp / initial_temp) ** (epoch / total_epochs)

def contrastive_loss(q, k, queue, epoch, total_epochs):
    """InfoNCE loss with temperature annealing"""
    temperature = temperature_schedule(epoch, total_epochs)
    N = q.shape[0]
    
    # Ensure normalization
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    queue = F.normalize(queue, dim=-1)
    
    # Positive similarity
    logits_pos = torch.sum(q * k, dim=1) / temperature
    
    # Negative similarity
    logits_neg = torch.matmul(q, queue) / temperature
    
    # Combine logits
    logits = torch.cat([logits_pos.unsqueeze(1), logits_neg], dim=1)
    
    # Labels (0 = positive pair)
    labels = torch.zeros(N, dtype=torch.long, device=q.device)
    
    # Cross entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss

#############################################
# Training and Evaluation Functions
#############################################

def generate_embeddings(encoder, dataloader, device, keep_grad=True):
    """Generate embeddings for the entire dataset"""
    # Store previous training state
    previous_state = encoder.training
    
    # Set to eval mode but keep track of gradients if needed
    encoder.eval()
    
    features_list = []
    labels_list = []
    
    # Use appropriate context manager based on keep_grad
    context_manager = torch.enable_grad() if keep_grad else torch.no_grad()
    with context_manager:
        for images, labels in dataloader:
            images = images.to(device)
            
            # Get encoder features (without projection head)
            features = encoder(images, return_before_head=True)
            
            # Important: Don't detach features if we need gradients
            features_list.append(features if keep_grad else features.detach())
            labels_list.append(labels)
    
    # Restore previous training state
    if previous_state:
        encoder.train()
    
    # Concatenate features from all batches
    all_features = torch.cat(features_list, dim=0)
    
    # Ensure labels are on the correct device
    all_labels = torch.cat(labels_list, dim=0).to(device)
    
    return all_features, all_labels

def evaluate_clustering(features, labels, n_clusters=10):
    """Evaluate clustering performance"""
    # Convert to numpy if tensor
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().detach().numpy()
    else:
        features_np = features
        
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().detach().numpy()
    else:
        labels_np = labels
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_np)
    
    # Calculate metrics
    nmi = normalized_mutual_info_score(labels_np, cluster_labels)
    ari = adjusted_rand_score(labels_np, cluster_labels)
    silhouette = silhouette_score(features_np, cluster_labels)
    db = davies_bouldin_score(features_np, cluster_labels)
    ch = calinski_harabasz_score(features_np, cluster_labels)
    
    # Transform davies_bouldin to inverse (higher is better) for consistency
    db_inv = 1.0 / (db + 1e-10)
    
    return {
        'nmi': nmi,
        'ari': ari,
        'silhouette': silhouette,
        'db_inv': db_inv,
        'db': db,
        'ch': ch,
    }

def train_and_evaluate(config, data_path, save_dir='model_weights'):
    """Main training and evaluation function"""
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data transformations based on config
    if config['use_advanced_aug']:
        transform = AdvancedAugmentation()
    else:
        transform = BasicAugmentation()
    
    # Create datasets and dataloaders
    dataset = FashionMNISTDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Dataset for embeddings (without heavy augmentations)
    embedding_dataset = EmbeddingDataset(data_path)
    embedding_dataloader = DataLoader(embedding_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize encoders
    encoder_q = create_model(config).to(device)
    encoder_k = create_model(config).to(device)
    
    # Copy weights from encoder_q to encoder_k
    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False
    
    # Setup optimizer
    optimizer = optim.SGD(
        encoder_q.parameters(),
        lr=config['lr'],
        momentum=0.9,
        weight_decay=0.0001
    )
    
    # Setup LR scheduler if using cosine annealing
    if config['use_cosine_lr']:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['epochs'], 
            eta_min=1e-5
        )
    
    # Setup mixed precision training
    scaler = amp.GradScaler()
    
    # Early stopping parameters
    patience = 30
    min_delta = 0.00001
    best_loss = float('inf')
    counter = 0
    
    # Initialize memory queue
    queue_size = config['queue_size']
    queue = torch.randn(128, queue_size).to(device)
    queue = F.normalize(queue, dim=0)
    queue_ptr = 0
    
    # Training loop
    start_time = time.time()
    for epoch in range(config['epochs']):
        running_loss = 0.0
        
        # Training
        encoder_q.train()
        for batch_idx, (q_images, k_images) in enumerate(dataloader):
            q_images = q_images.to(device)
            k_images = k_images.to(device)
            
            # Forward pass with mixed precision
            with amp.autocast():
                q = encoder_q(q_images)
                with torch.no_grad():
                    k = encoder_k(k_images)
                loss = contrastive_loss(q, k, queue, epoch, config['epochs'])
            
            # Backward and optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate if using cosine annealing
            if config['use_cosine_lr']:
                scheduler.step()
            
            # Momentum update of encoder_k
            m = 0.999
            for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)
            
            # Update queue
            batch_size = q.size(0)
            end_ptr = (queue_ptr + batch_size) % queue_size
            
            if end_ptr > queue_ptr:
                queue[:, queue_ptr:end_ptr] = k.T
            else:
                queue[:, queue_ptr:] = k.T[:, :queue_size - queue_ptr]
                queue[:, :end_ptr] = k.T[:, queue_size - queue_ptr:]
            queue_ptr = end_ptr
            
            # Update running loss
            running_loss += loss.item()
        
        # Calculate average loss
        avg_loss = running_loss / len(dataloader)
        
        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            counter = 0
            torch.save(encoder_q.state_dict(), save_path / f'encoder_q_best.pt')
        else:
            counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{config["epochs"]}], Loss: {avg_loss:.6f}')
            
            # Generate embeddings and evaluate clustering periodically
            encoder_q.eval()
            with torch.no_grad():
                features, labels = generate_embeddings(
                    encoder_q, embedding_dataloader, device, keep_grad=False
                )
            
            metrics = evaluate_clustering(features, labels)
            print(f"NMI: {metrics['nmi']:.4f}, ARI: {metrics['ari']:.4f}, Silhouette: {metrics['silhouette']:.4f}")
        
        # Check if we should stop early
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Calculate total training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Final evaluation
    print("\nGenerating final embeddings and evaluating clustering...")
    encoder_q.load_state_dict(torch.load(save_path / 'encoder_q_best.pt', map_location=device))
    encoder_q.eval()
    
    with torch.no_grad():
        final_features, final_labels = generate_embeddings(
            encoder_q, embedding_dataloader, device, keep_grad=False
        )
    
    # Convert to numpy for evaluation
    features_np = final_features.cpu().numpy()
    labels_np = final_labels.cpu().numpy()
    
    # Final clustering evaluation
    final_metrics = evaluate_clustering(features_np, labels_np)
    
    print(f"Final clustering evaluation:")
    print(f"NMI: {final_metrics['nmi']:.4f}")
    print(f"ARI: {final_metrics['ari']:.4f}")
    print(f"Silhouette Score: {final_metrics['silhouette']:.4f}")
    print(f"Davies-Bouldin: {final_metrics['db']:.4f}")
    
    return final_metrics

#############################################
# Main function to run the ablation study
#############################################

def run_ablation_study(data_path, cases=None):
    """Run the complete ablation study and print results in a table"""
    if cases is None:
        cases = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    
    results = {}
    
    print("\n============= Starting Ablation Study =============\n")
    
    for case in cases:
        print(f"\n========== Running experiment for case ({case}) ==========\n")
        config = get_config(case)
        
        # Print configuration
        print(f"Configuration for case ({case}):")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()
        
        # Create a unique directory for this experiment
        save_dir = f"model_weights/case_{case}"
        
        # Run the experiment
        metrics = train_and_evaluate(config, data_path, save_dir)
        
        # Store results
        results[case] = metrics
    
    # Print the results table
    print("\n============= Ablation Study Results =============\n")
    print("| Case | ConvNet | ResConn | MLP  | aug+ | cos  | size | epochs | NMI  | ARI  | Silhouette |")
    print("| ---- | ------- | ------- | ---- | ---- | ---- | ---- | ------ | ---- | ---- | ---------- |")
    
    for case in cases:
        config = get_config(case)
        m = results[case]
        print(f"| ({case})  | {'✓' if config['use_convnet'] else ' '}       | "
              f"{'✓' if config['use_resconn'] else ' '}       | "
              f"{'✓' if config['use_mlp'] else ' '}    | "
              f"{'✓' if config['use_advanced_aug'] else ' '}    | "
              f"{'✓' if config['use_cosine_lr'] else ' '}    | "
              f"{config['model_size']}   | "
              f"{config['epochs']}     | "
              f"{m['nmi']:.4f} | {m['ari']:.4f} | {m['silhouette']:.4f} |")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Fashion MNIST contrastive learning ablation study')
    parser.add_argument('--data_path', type=str, default='data/fashion-mnist_test.csv',
                        help='Path to the Fashion-MNIST CSV file')
    parser.add_argument('--cases', type=str, default='all',
                        help='Which cases to run, e.g., "a,b,c" or "all"')
    
    args = parser.parse_args()
    
    # Determine which cases to run
    if args.cases.lower() == 'all':
        cases = ['a', 'b', 'c', 'd', 'e', 'f']
    else:
        cases = args.cases.split(',')
    
    # Run the ablation study
    run_ablation_study(args.data_path, cases)