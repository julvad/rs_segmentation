# VAR
TS_FOLDER_PATH = 'data/512'
OUT_PRUNED_FOLDER_PATH = 'data/512_pruned_hkmeans_10'
PERCENT_KEEP = 0.1

#______________________________________________________________________________________________
import os
import numpy as np
import glob
import torch
from transformers import AutoModel
import rasterio
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import shutil
from math import ceil

# Load model and processor
model = AutoModel.from_pretrained("galeio-research/OceanSAR-1")

def hierarchical_kmeans(features, n_clusters_total, branch_factor=4, random_state=24):
    """
    Perform Hierarchical KMeans clustering.
    Recursively splits data until roughly n_clusters_total clusters are reached.

    Args:
        features (ndarray): Data matrix (N, D)
        n_clusters_total (int): Approximate total number of clusters desired
        branch_factor (int): Number of child clusters per split (default 4)
        random_state (int): Random seed

    Returns:
        centers (list of ndarray): List of cluster centers
        labels (ndarray): Cluster assignments (same length as features)
    """
    np.random.seed(random_state)

    N = len(features)
    labels = np.zeros(N, dtype=int)
    centers = []

    # Each node in the queue: (indices of samples, current depth)
    queue = [(np.arange(N), 0)]
    cluster_id = 0

    while queue and cluster_id < n_clusters_total:
        idxs, _ = queue.pop(0)
        subset = features[idxs]

        if len(subset) <= branch_factor or cluster_id + branch_factor > n_clusters_total:
            # Treat as leaf node
            center = np.mean(subset, axis=0)
            centers.append(center)
            labels[idxs] = cluster_id
            cluster_id += 1
            continue

        # Apply KMeans at this node
        kmeans = KMeans(n_clusters=branch_factor, random_state=random_state)
        kmeans.fit(subset)
        sub_labels = kmeans.labels_

        # Create child nodes for each branch
        for k in range(branch_factor):
            child_idxs = idxs[sub_labels == k]
            if len(child_idxs) == 0:
                continue
            queue.append((child_idxs, _ + 1))

    # Assign labels if unassigned (edge cases)
    unique_clusters = len(centers)
    if unique_clusters < n_clusters_total:
        # Pad with remaining averages
        remaining = n_clusters_total - unique_clusters
        for _ in range(remaining):
            rand_center = features[np.random.randint(0, N)]
            centers.append(rand_center)

    centers = np.array(centers[:n_clusters_total])
    return centers, labels


def prune_hkmeans(
        ts_folder_path: str,
        percent_keep: float,
        out_pruned_folder_path: str
    ):
    """
    Prune samples corresponding to the background class using hierarchical k-means.
    """
    for tt in ['train', 'test']:
        out_path = os.path.join(out_pruned_folder_path, tt)
        os.makedirs(out_path, exist_ok=True)

        list_samples = glob.glob(os.path.join(ts_folder_path, tt, 'images', '*.tif'))
        list_bg = [s for s in list_samples if 'bg' in os.path.basename(s)]
        list_imgs = [s for s in list_samples if 'bg' not in os.path.basename(s)]

        N = len(list_bg)
        print(f'Pruning {N} background files to {out_path}...')

        if N == 0:
            continue

        features_array = np.zeros((N, 2048), dtype=np.float32)
        paths = np.array(list_bg)

        for i, file_path in enumerate(paths):
            with rasterio.open(file_path) as src:
                band = src.read(1)

            tensor = torch.from_numpy(band).unsqueeze(0).unsqueeze(0).float()
            tensor = F.interpolate(tensor, size=(256, 256), mode='bilinear', align_corners=False)

            with torch.no_grad():
                outputs = model(tensor)
                features = outputs.pooler_output.squeeze()
                features_array[i] = features.cpu().numpy()

        N_new = int(N * percent_keep)
        print(f"Using Hierarchical K-Means to reduce {N} → {N_new} samples...")

        # Hierarchical K-Means clustering
        cluster_centers, labels = hierarchical_kmeans(features_array, n_clusters_total=N_new, branch_factor=4)
        
        # Keep the image closest to each cluster center
        keep_indices = []
        for i in range(len(cluster_centers)):
            cluster_features = features_array[labels == i]
            if len(cluster_features) == 0:
                continue
            center = cluster_centers[i].reshape(1, -1)
            distances = cdist(cluster_features, center, metric='euclidean')
            closest_idx = np.argmin(distances)
            original_idx = np.where(labels == i)[0][closest_idx]
            keep_indices.append(original_idx)

        keep_indices = np.array(keep_indices)
        keep_imgs = paths[keep_indices].tolist()
        keep_imgs.extend(list_imgs)

        print(f'{out_path}: Keeping {len(keep_imgs)} (with {len(list_imgs)} imgs) samples out of {len(list_samples)}')

        os.makedirs(os.path.join(out_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'labels'), exist_ok=True)

        for img_path in keep_imgs:
            filename = os.path.basename(img_path)
            dst_img = os.path.join(out_path, 'images', filename)
            shutil.copy2(img_path, dst_img)
            lbl_path = img_path.replace('images', 'labels')
            dst_lbl = dst_img.replace('images', 'labels')
            shutil.copy2(lbl_path, dst_lbl)
            print(f'Copied {filename} to pruned dataset.')

if __name__ == '__main__':
    prune_hkmeans(
        ts_folder_path=TS_FOLDER_PATH,
        percent_keep=PERCENT_KEEP,
        out_pruned_folder_path=OUT_PRUNED_FOLDER_PATH
    )
