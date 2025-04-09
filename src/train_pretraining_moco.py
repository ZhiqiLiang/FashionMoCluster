import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import random
import torch.cuda.amp as amp
import cv2
import argparse

# Step1: dataset.py
class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        
        # Calculate the pixel mean for the entire dataset
        images = self.data.iloc[:, 1:].values.reshape(-1, 28, 28).astype('float32')
        self.pixel_mean = images.mean()
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert to numpy array and subtract the mean
        image = self.data.iloc[idx, 1:].values.reshape(28, 28).astype('float32')
        image = image - self.pixel_mean  # Subtract pixel mean
        
        # Normalize values to [0, 255] range
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype('uint8')
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            q, k = self.transform(image)
            return q, k
        return image

# Step2: model.py
class ResidualBlock(nn.Module):
    expansion = 1  # Expansion factor for output channels relative to input channels

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels) 
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

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
        self.layer1 = self._make_layer(residual_block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(residual_block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(residual_block, 64, num_blocks[2], stride=2)
        
        self.linear = nn.Linear(64 * 7 * 7, 128)
        if projection:
            self.projector = ProjectionHead()

    def _make_layer(self, residual_block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(residual_block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * residual_block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.normalize(out, dim=-1)  # Normalize at the end
        
        if hasattr(self, 'projector') and self.projection:
            out = self.projector(out)
        
        return out

def ResNet32(projection=True): 
    return ResNet(ResidualBlock, [5, 5, 5], projection)

# Temperature Annealing
def temperature_schedule(epoch, total_epochs, initial_temp=0.07, final_temp=0.01):
    return initial_temp * (final_temp / initial_temp) ** (epoch / total_epochs)

def contrastive_loss(q, k, queue, epoch, total_epochs):
    temperature = temperature_schedule(epoch, total_epochs)
    N = q.shape[0]
    
    # Normalize
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    queue = F.normalize(queue, dim=-1)
    
    # Positive similarity
    logits_pos = torch.sum(q * k, dim=1) / temperature
    
    # Negative similarity
    logits_neg = torch.matmul(q, queue) / temperature
    
    # Combine logits
    logits = torch.cat([logits_pos.unsqueeze(1), logits_neg], dim=1)
    
    # Labels
    labels = torch.zeros(N, dtype=torch.long, device=q.device)
    
    # Cross entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss

# Step3: main.py
class GrayscaleSimCLRAugmentation:
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

def main(args):
    # Setup GPU environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Create dataset instance
    transform = GrayscaleSimCLRAugmentation()
    dataset = FashionMNISTDataset(args.dataset_path, transform=transform)
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize encoder
    encoder_q = ResNet32().to(device)
    encoder_k = ResNet32().to(device)
    # Copy weights
    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
        param_k.data.copy_(param_q.data)  # copy weights
        param_k.requires_grad = False  # not update by gradient

    # Training parameters
    num_epochs = args.num_epochs

    # Define SGD optimizer
    optimizer = optim.SGD(encoder_q.parameters(), 
                          lr=args.lr,  # initial learning rate
                          momentum=0.9,  # momentum
                          weight_decay=0.0001)  # weight decay
    # Cosine Annealing Learning Rate
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # Early stopping strategy
    patience = args.patience
    min_delta = 0.00001  # minimum loss decrease for early stopping
    best_loss = float('inf')
    counter = 0

    # Save checkpoint frequency
    save_every_n_epochs = args.save_every_n_epochs

    # Create model save path
    model_save_path = Path(args.model_save_path)
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Initialize queue
    queue_size = 8192
    queue = torch.randn(128, queue_size).to(device)  # initialize random tensor
    queue = F.normalize(queue, dim=0)  # L2 normalize
    queue_ptr = 0  # pointer to current insertion position

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (q_images, k_images) in enumerate(dataloader):
            q_images = q_images.to(device)
            k_images = k_images.to(device)

            # Forward pass
            with amp.autocast():
                q = encoder_q(q_images)
                with torch.no_grad():
                    k = encoder_k(k_images)
                # Calculate loss
                loss = contrastive_loss(q, k, queue, epoch, num_epochs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Momentum update for encoder_k
            m = 0.999
            for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)

            # Update queue
            batch_size = q.size(0)
            end_ptr = (queue_ptr + batch_size) % queue_size  # calculate new pointer position
            if end_ptr > queue_ptr:  # when end_ptr is greater than queue_ptr
                queue[:, queue_ptr:end_ptr] = k.T  # directly fill between queue_ptr and end_ptr
            else:  # when end_ptr is less than or equal to queue_ptr
                queue[:, queue_ptr:] = k.T[:, :queue_size - queue_ptr]  # update the part from queue_ptr to the end
                queue[:, :end_ptr] = k.T[:, queue_size - queue_ptr:]  # fill remaining part from the start
            queue_ptr = end_ptr  # update pointer

            # Accumulate loss
            running_loss += loss.item()

        # Calculate average loss
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

        # Early stopping strategy
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            counter = 0
            torch.save(encoder_q.state_dict(), model_save_path / 'encoder_q_moco_best.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Save checkpoint
        if (epoch + 1) % save_every_n_epochs == 0:
            torch.save(encoder_q.state_dict(), model_save_path / f'encoder_q_moco_epoch_{epoch+1}.pt')

    # Save final model
    torch.save(encoder_q.state_dict(), model_save_path / 'encoder_q_moco_final.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training MoCo with Grayscale Augmentation")
    parser.add_argument('--dataset_path', type=str, default='data/fashion-mnist_test.csv', help="Path to the dataset CSV file")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=300, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.03, help="Learning rate")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience")
    parser.add_argument('--save_every_n_epochs', type=int, default=20, help="Checkpoint save frequency")
    parser.add_argument('--model_save_path', type=str, default='model_weights', help="Path to save model checkpoints")
    
    args = parser.parse_args()
    main(args)