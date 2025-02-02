import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import umap
import matplotlib.pyplot as plt
from collections import Counter
import joblib
from PIL import Image

# ----------------------------
# 1. Define Custom Dataset
# ----------------------------

class SeismicDataset(Dataset):
    def __init__(self, root_dir, transform=None, mat_key='image'):
        """
        Args:
            root_dir (string): Directory with all the seismic .mat files, organized in subdirectories per class.
            transform (callable, optional): Optional transform to be applied on a sample.
            mat_key (string): The key in the .mat file that contains the image data.
        """
        self.mat_paths = []
        self.labels = []
        self.transform = transform
        self.mat_key = mat_key

        # Assuming each subdirectory in root_dir corresponds to a class
        for label in os.listdir(root_dir):
            label_dir=root_dir+'//'+label
            for filename in os.listdir(label_dir):
                if filename.lower().endswith('.mat'):
                    self.mat_paths.append(os.path.join(label_dir, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.mat_paths)

    def __getitem__(self, idx):
        mat_path = self.mat_paths[idx]
        label = self.labels[idx]

        try:
            mat_contents = loadmat(mat_path)
            if self.mat_key not in mat_contents:
                raise KeyError(f"Key '{self.mat_key}' not found in {mat_path}. Available keys: {list(mat_contents.keys())}")
            image_array = mat_contents[self.mat_key]

            # Handle image dimensions
            if image_array.ndim == 2:
                # Grayscale image, convert to 3-channel by replicating
                image_array = np.stack([image_array]*3, axis=-1)
            elif image_array.ndim == 3 and image_array.shape[2] == 1:
                # Single-channel image, replicate to 3 channels
                image_array = np.concatenate([image_array]*3, axis=2)
            elif image_array.ndim == 3 and image_array.shape[2] == 3:
                # Already 3-channel
                pass
            else:
                raise ValueError(f"Unexpected image dimensions {image_array.shape} in {mat_path}")

            # Normalize the image array to [0, 255] and convert to uint8
            image_min, image_max = image_array.min(), image_array.max()
            if image_max - image_min > 0:
                image_array = (image_array - image_min) / (image_max - image_min) * 255.0
            else:
                image_array = np.zeros_like(image_array)
            image_array = image_array.astype(np.uint8)
            # Convert to PIL Image
            image = transforms.ToPILImage()(image_array)

        except Exception as e:
            print(f"Error loading or processing {mat_path}: {e}")
            # Return a black image in case of error
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

# ----------------------------
# 2. Function to Extract Features and Save Batches
# ----------------------------

def extract_and_save_features(dataloader, model, device, save_dir):
    """
    Extract features using the provided model and save each batch of features and labels.

    Args:
        dataloader (DataLoader): PyTorch DataLoader for the dataset.
        model (torch.nn.Module): Feature extractor model.
        device (torch.device): Device to perform computations on.
        save_dir (string): Directory to save the feature batches.
    
    Returns:
        None
    """
        
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, (inputs, batch_labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten
            features = outputs.cpu().numpy()
            labels = batch_labels
            # Save features and labels for the current batch
            feature_path = os.path.join(save_dir, f'features_batch_{batch_idx}.npy')
            label_path = os.path.join(save_dir, f'labels_batch_{batch_idx}.npy')
            np.save(feature_path, features)
            np.save(label_path, labels)

            print(f'Saved batch {batch_idx}: Features at {feature_path}, Labels at {label_path}')

# ----------------------------
# 3. Function to Load Saved Features and Labels
# ----------------------------

def load_saved_features(save_dir):
    """
    Load saved feature and label batches from the specified directory.

    Args:
        save_dir (string): Directory where feature and label batches are saved.

    Returns:
        all_features (numpy.ndarray): Concatenated feature array.
        all_labels (list): List of all labels.
    """
    all_features = []
    all_labels = []

    # List and sort feature and label files
    feature_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith('features_batch_') and f.endswith('.npy')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    label_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith('labels_batch_') and f.endswith('.npy')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    # Load each batch
    for feature_file, label_file in zip(feature_files, label_files):
        feature_path = os.path.join(save_dir, feature_file)
        label_path = os.path.join(save_dir, label_file)
        features = np.load(feature_path)
        labels = np.load(label_path, allow_pickle=True)
        all_features.append(features)
        all_labels.extend(labels)

        print(f'Loaded batch {feature_file} and {label_file}')

    # Concatenate all features
    all_features = np.vstack(all_features)
    print(f'Total features shape after loading all batches: {all_features.shape}')
    print(f'Total number of labels: {len(all_labels)}')

    return all_features, all_labels

# ----------------------------
# 5. Main Execution Block for Training and Saving Models
# ----------------------------

def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress specific warnings

    # ----------------------------
    # 1. Set Up Data Transformations
    # ----------------------------

    # Define image transformations: resizing, normalization as per pre-trained model requirements
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])

    # ----------------------------
    # 2. Load Dataset
    # ----------------------------

    # Replace 'path_to_landmass_dataset' with the actual path to your dataset
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' does not exist. Please check the path.")
        return

    # Specify the key in the .mat files that contains the image data
    mat_key = 'img'  # Change this to the actual key in your .mat files

    dataset = SeismicDataset(root_dir=dataset_path, transform=transform, mat_key=mat_key)
    if len(dataset) == 0:
        print("No .mat files found in the dataset. Please check the directory structure and file extensions.")
        return

    # Define DataLoader parameters
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # ----------------------------
    # 3. Feature Extraction with Pre-trained CNN
    # ----------------------------

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load pre-trained ResNet50 model with updated weights parameter
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet50 = resnet50.to(device)
    resnet50.eval()

    # Remove the final classification layer
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])

    # Define directory to save feature batches
    save_dir = 'saved_features'
    os.makedirs(save_dir, exist_ok=True)

    print('Extracting and saving features...')
    extract_and_save_features(dataloader, feature_extractor, device, save_dir)
    print('Feature extraction and saving completed.')

    # ----------------------------
    # 4. Load Saved Features for Clustering
    # ----------------------------

    all_features, true_labels = load_saved_features(save_dir)

    # ----------------------------
    # 5. Dimensionality Reduction with UMAP
    # ----------------------------

    print('Performing UMAP dimensionality reduction...')
    reducer = umap.UMAP(n_components=50, random_state=42)
    features_reduced = reducer.fit_transform(all_features)
    print(f'Reduced features shape: {features_reduced.shape}')

    # Save the UMAP reducer
    joblib.dump(reducer, 'umap_reducer.joblib')
    print("UMAP reducer saved as 'umap_reducer.joblib'.")

    # ----------------------------
    # 6. Clustering with KMeans
    # ----------------------------

    num_clusters = 3  # As per the four geological classes
    print('Clustering with KMeans...')
    kmeans = KMeans(n_clusters=num_clusters, n_init=20, random_state=42)
    cluster_assignments = kmeans.fit_predict(features_reduced)
    print('Clustering completed.')

    # Save the KMeans model
    joblib.dump(kmeans, 'kmeans_model.joblib')
    print("KMeans model saved as 'kmeans_model.joblib'.")

    # ----------------------------
    # 7. Evaluation
    # ----------------------------

    # Encode true labels as integers
    label_set = sorted(set(true_labels))
    label_to_int = {label: idx for idx, label in enumerate(label_set)}
    true_labels_int = np.array([label_to_int[label] for label in true_labels])

    # Calculate Adjusted Rand Index
    ari = adjusted_rand_score(true_labels_int, cluster_assignments)
    print(f'Adjusted Rand Index (ARI): {ari:.4f}')

    # Analyze cluster balance
    cluster_counts = Counter(cluster_assignments)
    print('Cluster distribution:')
    for cluster, count in cluster_counts.items():
        print(f'  Cluster {cluster}: {count} samples')

    # ----------------------------
    # 8. Map Clusters to Labels
    # ----------------------------

    # Create a mapping from cluster to label based on majority voting
    cluster_to_label = {}
    for cluster in range(num_clusters):
        # Find indices of samples in the current cluster
        indices = np.where(cluster_assignments == cluster)[0]
        # Get the true labels of these samples
        cluster_labels = true_labels_int[indices]
        # Find the most common label in the cluster
        if len(cluster_labels) > 0:
            most_common_label = Counter(cluster_labels).most_common(1)[0][0]
            # Map to the actual label name
            cluster_to_label[cluster] = label_set[most_common_label]
        else:
            cluster_to_label[cluster] = 'Unknown'
        print(f'Cluster {cluster} is mapped to label "{cluster_to_label[cluster]}"')

    # Save the cluster_to_label mapping
    np.save('cluster_to_label_mapping.npy', cluster_to_label)
    print("Cluster to label mapping saved as 'cluster_to_label_mapping.npy'.")

    # ----------------------------
    # 9. Visualization with UMAP
    # ----------------------------

    # Further reduce to 2D for visualization
    print('Performing UMAP for visualization...')
    reducer_2d = umap.UMAP(n_components=2, random_state=42)
    features_2d = reducer_2d.fit_transform(features_reduced)
    print(f'2D UMAP features shape: {features_2d.shape}')

    # Save the 2D UMAP reducer
    joblib.dump(reducer_2d, 'umap_reducer_2d.joblib')
    print("2D UMAP reducer saved as 'umap_reducer_2d.joblib'.")

    # Plot UMAP Projection Colored by Cluster Assignments
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=cluster_assignments, cmap='viridis', s=10)
    plt.title('UMAP Projection of Seismic Images Colored by KMeans Cluster Assignment')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('umap_kmeans_clusters.png')
    plt.close()
    print("UMAP plot colored by KMeans clusters saved as 'umap_kmeans_clusters.png'.")

    # Plot UMAP Projection Colored by True Labels
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=true_labels_int, cmap='tab10', s=10)
    plt.title('UMAP Projection of Seismic Images Colored by True Labels')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.colorbar(scatter, ticks=range(num_clusters), label='True Label')
    plt.savefig('umap_true_labels.png')
    plt.close()
    print("UMAP plot colored by true labels saved as 'umap_true_labels.png'.")
    # Save the feature extractor model
    torch.save(feature_extractor.state_dict(), 'feature_extractor.pth')
    print("Feature extractor model saved as 'feature_extractor.pth'.")

    # Save clustering results
    np.save('cluster_assignments.npy', cluster_assignments)
    print("KMeans cluster assignments saved as 'cluster_assignments.npy'.")

    # Save UMAP reducer (already saved earlier)

    # ----------------------------
    # End of Pipeline
    # ----------------------------
import pandas as pd
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load models and mappings
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet50 = resnet50.to(device)
    resnet50.eval()
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
    feature_extractor.load_state_dict(torch.load('feature_extractor.pth', map_location=device))
    feature_extractor.eval()
    
    reducer = joblib.load('umap_reducer.joblib')
    kmeans = joblib.load('kmeans_model.joblib')
    cluster_to_label = np.load('cluster_to_label_mapping.npy', allow_pickle=True).item()

    # Data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    new_data_path = "test_dataset"
    new_dataset = SeismicDataset(root_dir=new_data_path, transform=transform, mat_key='img')
    new_dataloader = DataLoader(new_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Feature extraction
    new_features = []
    with torch.no_grad():
        for inputs, _ in new_dataloader:
            inputs = inputs.to(device)
            features = feature_extractor(inputs).view(inputs.size(0), -1)
            new_features.append(features.cpu().numpy())
    new_features = np.vstack(new_features)

    # Dimensionality reduction and clustering
    new_features_reduced = reducer.transform(new_features)
    cluster_assignments = kmeans.predict(new_features_reduced)
    predicted_labels = [cluster_to_label[cluster] for cluster in cluster_assignments]

    # Get true labels and file paths
    true_labels = new_dataset.labels
    file_paths = new_dataset.mat_paths

    # Calculate metrics per class
    classes = sorted(set(true_labels))
    metrics = []
    for class_name in classes:
        y_true = np.array([1 if label == class_name else 0 for label in true_labels])
        y_pred = np.array([1 if label == class_name else 0 for label in predicted_labels])
        
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        metrics.append({
            'Class': class_name,
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'Precision': TP/(TP+FP) if (TP+FP)!=0 else 0,
            'Recall': TP/(TP+FN) if (TP+FN)!=0 else 0
        })

    # Print and save results
    metrics_df = pd.DataFrame(metrics)
    print("\nClassification Metrics:")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv('classification_metrics.csv', index=False)

    results_df = pd.DataFrame({
        'File': [os.path.basename(p) for p in file_paths],
        'True_Label': true_labels,
        'Predicted_Label': predicted_labels
    })
    results_df.to_csv('predictions.csv', index=False)
    print("Results saved!")
    class_distribution = Counter(true_labels)
    print("\nClass Distribution in Test Data:")
    for cls, count in class_distribution.items():
        print(f"{cls}: {count} samples ({count/len(true_labels):.1%})")

    # Enhanced metric calculation
    metrics = []
    for class_name in classes:
        y_true = np.array([label == class_name for label in true_labels])
        y_pred = np.array([label == class_name for label in predicted_labels])
        
        TP = np.sum(y_true & y_pred)
        FP = np.sum(~y_true & y_pred)
        TN = np.sum(~y_true & ~y_pred)
        FN = np.sum(y_true & ~y_pred)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'Class': class_name,
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'Precision': f"{precision:.1%}",
            'Recall': f"{recall:.1%}",
            'F1-Score': f"{f1:.1%}",
            'Support': class_distribution[class_name]
        })

    # Print formatted metrics
    print("\nEnhanced Classification Report:")
    print(pd.DataFrame(metrics).to_string(index=False))
if __name__=='__main__':
    main()
    