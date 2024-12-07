import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class ASLGestureDataset(Dataset):
    def __init__(self, root_dir, json_path, transform=None):
        """
        Custom dataset for ASL gesture videos
        
        Args:
        - root_dir (str): Path to the frames directory
        - json_path (str): Path to the JSON file with bounding box and frame information
        - transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.gestures = os.listdir(root_dir)
        
        # Load bounding box and frame information
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.gestures)
    
    def __getitem__(self, idx):
        gesture = self.gestures[idx]
        gesture_path = os.path.join(self.root_dir, gesture)
        
        # Find all instances for this gesture
        instances = [d for d in os.listdir(gesture_path) if os.path.isdir(os.path.join(gesture_path, d))]
        
        if not instances:
            return None
        
        # Randomly select an instance
        instance = np.random.choice(instances)
        instance_path = os.path.join(gesture_path, instance)
        
        # Get frames
        frames = sorted([f for f in os.listdir(instance_path) if f.endswith('.jpg')])
        
        # Get annotation details
        annotation = self.annotations.get(f"{gesture}/{instance}", {})
        frame_start = annotation.get('frame_start', 0)
        frame_end = annotation.get('frame_end', len(frames))
        
        # If frame_end is -1, use all frames
        if frame_end == -1:
            frame_end = len(frames)
        
        # Select frames within the gesture interval
        gesture_frames = frames[frame_start:frame_end]
        
        # If not enough frames, pad or repeat
        if len(gesture_frames) < 2:
            gesture_frames = gesture_frames * 2
        
        # Select two different frames for contrastive learning
        frame1, frame2 = np.random.choice(gesture_frames, 2, replace=False)
        
        # Load and transform frames
        img1 = self.transform(Image.open(os.path.join(instance_path, frame1)))
        img2 = self.transform(Image.open(os.path.join(instance_path, frame2)))
        
        return img1, img2

class ContrastiveNetwork(nn.Module):
    def __init__(self, backbone='resnet18', projection_dim=128):
        """
        Contrastive learning network with a projection head
        
        Args:
        - backbone (str): Base CNN architecture
        - projection_dim (int): Dimension of the projection head
        """
        super().__init__()
        
        # Use a pre-trained backbone
        if backbone == 'resnet18':
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
    
    def forward(self, x):
        """
        Forward pass with feature extraction and projection
        
        Args:
        - x (tensor): Input image
        
        Returns:
        - Projected feature representation
        """
        features = self.backbone(x)
        return self.projector(features)

def contrastive_loss(z1, z2, temperature=0.5):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
    
    Args:
    - z1, z2 (tensor): Projected features from two views of an image
    - temperature (float): Temperature scaling parameter
    
    Returns:
    - Contrastive loss
    """
    batch_size = z1.size(0)
    
    # Compute similarity matrix
    z = torch.cat([z1, z2], dim=0)
    similarity = torch.mm(z, z.t()) / temperature
    
    # Create mask to ignore self-similarities
    mask = torch.eye(batch_size * 2, device=z.device).bool()
    similarity.masked_fill_(mask, float('-inf'))
    
    # Compute softmax
    log_prob = torch.log_softmax(similarity, dim=1)
    
    # Compute loss
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])
    loss = nn.CrossEntropyLoss()(log_prob, labels)
    
    return loss

def train_self_supervised(model, dataloader, optimizer, device, epochs=10):
    """
    Self-supervised training loop
    
    Args:
    - model (nn.Module): Contrastive learning model
    - dataloader (DataLoader): Dataset loader
    - optimizer (Optimizer): Optimization algorithm
    - device (torch.device): Computing device
    - epochs (int): Number of training epochs
    """
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in dataloader:
            # Skip empty batches
            if batch is None:
                continue
            
            img1, img2 = batch
            img1, img2 = img1.to(device), img2.to(device)
            
            optimizer.zero_grad()
            
            # Get projected representations
            z1 = model(img1)
            z2 = model(img2)
            
            # Compute contrastive loss
            loss = contrastive_loss(z1, z2)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
    
    return model

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    epochs = 10
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset and DataLoader
    dataset = ASLGestureDataset(
        root_dir='frames', 
        json_path='gesture_annotations.json'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Model initialization
    model = ContrastiveNetwork()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    trained_model = train_self_supervised(
        model, 
        dataloader, 
        optimizer, 
        device, 
        epochs
    )
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'asl_self_supervised_model.pth')

if __name__ == '__main__':
    main()