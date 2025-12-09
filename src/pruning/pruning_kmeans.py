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

# Load model and processor
model = AutoModel.from_pretrained("galeio-research/OceanSAR-1")

def prune_kmeans(
        ts_folder_path:str,
        keep_frac:float,
        out_pruned_folder_path:str
    ):
        """
        Prune samples corresponding to the background class from features and labels.
        """
        assert 0<= keep_frac <=1
        for tt in ['train','test']:
            out_path = os.path.join(out_pruned_folder_path,tt)
            os.makedirs(out_path)
            list_samples= glob.glob(os.path.join(ts_folder_path,tt,'images','*.tif'))
            list_bg = []
            list_imgs = []
            for sample_path in list_samples:
                filename = os.path.basename(sample_path)
                if 'bg' in filename:
                    list_bg.append(sample_path)
                else:
                    list_imgs.append(sample_path)
            N = len(list_bg)
            print(f'Pruning {N} files from to {out_path} ({keep_frac*100}%)')

            features_array = np.zeros((N, 2048), dtype=np.float32)
            features_dict = {
                "paths": np.array(list_bg),
                "features": features_array
            }

            for i, file_path in enumerate(features_dict['paths']):
                # Read the image band
                with rasterio.open(file_path) as src:
                    band = src.read(1)

                # Convert to tensor and resize to 256 (oceansar input)
                tensor = torch.from_numpy(band).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
                tensor = F.interpolate(tensor, size=(256, 256), mode='bilinear', align_corners=False)

                # Extract features
                with torch.no_grad():
                    outputs = model(tensor)  # shape: [1, 2048, 1, 1]
                    features = outputs.pooler_output.squeeze()  # shape: [2048]
                    features_dict['features'][i] = features.cpu().numpy()
            
            N_new = int(N * keep_frac)  # number of images to keep

            # KMeans clustering
            kmeans = KMeans(n_clusters=N_new, random_state=24)
            kmeans.fit(features_array)
            cluster_centers = kmeans.cluster_centers_
            labels = kmeans.labels_

            # Keep the image closest to each cluster center
            keep_indices = []
            for i in range(N_new):
                cluster_features = features_array[labels == i]
                if len(cluster_features) == 0:
                    continue
                center = cluster_centers[i].reshape(1, -1)
                distances = cdist(cluster_features, center, metric='euclidean')
                closest_idx = np.argmin(distances)
                # Map back to original indices
                original_idx = np.where(labels == i)[0][closest_idx]
                keep_indices.append(original_idx)

            keep_indices = np.array(keep_indices)
            keep_imgs = features_dict['paths'][keep_indices].tolist()  # convert to list

            keep_imgs.extend(list_imgs)  # add non-bg images
            print(f'{out_path}: Keeping {len(keep_imgs)} (with {len(list_imgs)} imgs) samples out of {len(list_samples)}')

            os.makedirs(os.path.join(out_path,'images'))
            os.makedirs(os.path.join(out_path,'labels'))
            for img_path in keep_imgs:
                # Preserve the original filename
                filename = os.path.basename(img_path)
                dst_path = os.path.join(out_path,'images',filename)
                shutil.copy2(img_path, dst_path)
                lbl_path = img_path.replace('images','labels')
                dst_lbl_path = dst_path.replace('images','labels')
                shutil.copy2(lbl_path, dst_lbl_path)
                print(f'Copied {filename} to pruned dataset.')

if __name__ == '__main__':
    # VAR
    TS_FOLDER_PATH = 'data/512'
    OUT_PRUNED_FOLDER_PATH = 'data/512_pruned'
    KEEP_FRAC = 0.4
    prune_kmeans(
        ts_folder_path=TS_FOLDER_PATH,
        keep_frac=KEEP_FRAC,
        out_pruned_folder_path=OUT_PRUNED_FOLDER_PATH
    )