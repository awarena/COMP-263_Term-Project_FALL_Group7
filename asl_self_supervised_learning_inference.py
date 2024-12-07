import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

class ASLFeatureExtractor:
    def __init__(self, model_path, backbone='resnet18', device=None):
        """
        Feature extractor for ASL gesture representations
        
        Args:
        - model_path (str): Path to the saved model weights
        - backbone (str): Backbone network used during training
        - device (torch.device, optional): Compute device
        """
        # Use the same ContrastiveNetwork from training script
        from asl_self_supervised_learning import ContrastiveNetwork
        
        # Device configuration
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = ContrastiveNetwork(backbone=backbone)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Default transform (same as during training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path):
        """
        Extract features from a single image
        
        Args:
        - image_path (str): Path to the input image
        
        Returns:
        - Numpy array of extracted features
        """
        # Load and transform image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.backbone(img_tensor)
            projected_features = self.model.projector(features)
        
        return features.cpu().numpy(), projected_features.cpu().numpy()
    
    def extract_gesture_features(self, gesture_folder):
        """
        Extract features for all frames in a gesture instance
        
        Args:
        - gesture_folder (str): Path to gesture instance folder
        
        Returns:
        - Dict with frame names and their features
        """
        frame_features = {}
        
        # Sort frames
        frames = sorted([f for f in os.listdir(gesture_folder) if f.endswith('.jpg')])
        
        for frame in frames:
            frame_path = os.path.join(gesture_folder, frame)
            backbone_features, projected_features = self.extract_features(frame_path)
            
            frame_features[frame] = {
                'backbone_features': backbone_features.squeeze(),
                'projected_features': projected_features.squeeze()
            }
        
        return frame_features
    
    def compute_gesture_representation(self, gesture_folder):
        """
        Compute aggregate representation for an entire gesture
        
        Args:
        - gesture_folder (str): Path to gesture instance folder
        
        Returns:
        - Aggregate feature representation
        """
        frame_features = self.extract_gesture_features(gesture_folder)
        
        # Aggregate method: Mean pooling of projected features
        all_features = np.array([
            features['projected_features'] 
            for features in frame_features.values()
        ])
        
        return {
            'mean_features': np.mean(all_features, axis=0),
            'std_features': np.std(all_features, axis=0)
        }

def main():
    # Initialize feature extractor
    extractor = ASLFeatureExtractor(
        model_path='asl_self_supervised_model.pth'
    )
    
    # Example usage: Extract features for a specific gesture instance
    gesture_folder = 'frames/africa/01383'
    
    # Extract frame-level features
    frame_features = extractor.extract_gesture_features(gesture_folder)
    print("Frame-level features extracted:")
    for frame, features in frame_features.items():
        print(f"{frame}: Backbone features shape {features['backbone_features'].shape}")
    
    # Compute aggregate gesture representation
    gesture_rep = extractor.compute_gesture_representation(gesture_folder)
    print("\nGesture Representation:")
    print("Mean Features Shape:", gesture_rep['mean_features'].shape)
    print("Std Features Shape:", gesture_rep['std_features'].shape)
    
    # Similarity comparison example
    def compute_cosine_similarity(rep1, rep2):
        """Compute cosine similarity between two feature representations"""
        return np.dot(rep1, rep2) / (np.linalg.norm(rep1) * np.linalg.norm(rep2))
    
    # Compare two gesture instances
    another_gesture_folder = 'frames/africa/01387'
    another_gesture_rep = extractor.compute_gesture_representation(another_gesture_folder)
    
    similarity = compute_cosine_similarity(
        gesture_rep['mean_features'], 
        another_gesture_rep['mean_features']
    )
    print(f"\nSimilarity between gestures: {similarity}")

if __name__ == '__main__':
    main()