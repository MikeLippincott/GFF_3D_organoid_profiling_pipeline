from typing import Optional, Tuple

import numpy as np
import tifffile  # For reading microscope image stacks
import torch
from scipy import ndimage
from skimage import filters, morphology


class MicroscopeVolumeToPointCloud:
    """
    Convert 3D microscope volume images to point clouds for PointBERT.

    PointBERT expects point clouds with shape (N, 3) or (N, 6) where:
    - N is the number of points
    - First 3 columns are XYZ coordinates
    - Optional last 3 columns are features (e.g., intensity, RGB, normals)
    """

    def __init__(
        self,
        target_points: int = 2048,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        intensity_threshold: Optional[float] = None,
    ):
        """
        Args:
            target_points: Number of points to sample (PointBERT typically uses 1024-8192)
            voxel_size: Physical size of voxels (z, y, x) in micrometers
            intensity_threshold: Threshold for segmentation (None for auto Otsu)
        """
        self.target_points = target_points
        self.voxel_size = np.array(voxel_size)
        self.intensity_threshold = intensity_threshold

    def load_volume(self, filepath: str) -> np.ndarray:
        """Load 3D microscope volume from various formats."""
        # Supports TIFF stacks, which are common in microscopy
        volume = tifffile.imread(filepath)

        # Ensure 3D
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]

        return volume

    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess: denoise and normalize."""
        # Gaussian smoothing to reduce noise
        volume_smooth = ndimage.gaussian_filter(volume, sigma=1.0)

        # Normalize to [0, 1]
        volume_norm = (volume_smooth - volume_smooth.min()) / (
            volume_smooth.max() - volume_smooth.min()
        )

        return volume_norm

    def segment_volume(self, volume: np.ndarray) -> np.ndarray:
        """Segment foreground from background."""
        if self.intensity_threshold is None:
            # Automatic Otsu thresholding
            threshold = filters.threshold_otsu(volume)
        else:
            threshold = self.intensity_threshold

        # Binary mask
        mask = volume > threshold

        # Morphological operations to clean up
        mask = morphology.remove_small_objects(mask, min_size=100)
        mask = morphology.binary_closing(mask, footprint=morphology.ball(2))

        return mask

    def extract_points(
        self, volume: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract point coordinates and features from volume."""
        # Get coordinates of foreground voxels
        z_coords, y_coords, x_coords = np.where(mask)

        # Scale by voxel size to get physical coordinates
        coords = np.stack(
            [
                z_coords * self.voxel_size[0],
                y_coords * self.voxel_size[1],
                x_coords * self.voxel_size[2],
            ],
            axis=1,
        )

        # Extract intensity features
        intensities = volume[z_coords, y_coords, x_coords]

        # Optional: compute local features (gradient magnitude)
        gradient = np.stack(np.gradient(volume), axis=0)
        gradient_mag = np.linalg.norm(gradient, axis=0)
        gradients = gradient_mag[z_coords, y_coords, x_coords]

        # Stack features
        features = np.stack([intensities, gradients], axis=1)

        return coords, features

    def sample_points(
        self, coords: np.ndarray, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample to target number of points."""
        n_points = coords.shape[0]

        if n_points > self.target_points:
            # Random sampling
            indices = np.random.choice(n_points, self.target_points, replace=False)
        else:
            # Upsample with replacement if needed
            indices = np.random.choice(n_points, self.target_points, replace=True)

        return coords[indices], features[indices]

    def normalize_pointcloud(self, coords: np.ndarray) -> np.ndarray:
        """Normalize point cloud to unit sphere (common for PointBERT)."""
        # Center at origin
        centroid = coords.mean(axis=0)
        coords_centered = coords - centroid

        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(coords_centered, axis=1))
        coords_normalized = coords_centered / max_dist

        return coords_normalized

    def convert(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        normalize: bool = True,
    ) -> dict:
        """
        Full pipeline: volume -> point cloud.

        Returns:
            dict with keys:
                - 'coords': (N, 3) z,y,x coordinates
                - 'features': (N, F) features
                - 'points': (N, 3+F) combined for PointBERT
        """
        # Preprocess
        volume_proc = self.preprocess_volume(volume)

        # Segment
        if mask is None:
            mask = self.segment_volume(volume_proc)
        else:
            volume_proc[mask == 0] = 0  # Zero out background if mask provided

        # Extract points
        coords, features = self.extract_points(volume_proc, mask)

        # Sample
        coords, features = self.sample_points(coords, features)

        # Normalize coordinates
        if normalize:
            coords = self.normalize_pointcloud(coords)

        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        # Combine for PointBERT input
        points = np.concatenate([coords, features], axis=1)

        return {"coords": coords, "features": features, "points": points}


# Example usage with PointBERT
def prepare_for_pointbert(pointcloud: dict, use_features: bool = True) -> torch.Tensor:
    """
    Prepare point cloud tensor for PointBERT input.

    Args:
        pointcloud: Output from MicroscopeVolumeToPointCloud.convert()
        use_features: Whether to include features (coords+features) or just coords

    Returns:
        torch.Tensor of shape (N, 3) or (N, 3+F)
    """
    if use_features:
        points = pointcloud["points"]
    else:
        points = pointcloud["coords"]

    # Convert to torch tensor
    points_tensor = torch.from_numpy(points).float()

    return points_tensor


# Example: Complete workflow
if __name__ == "__main__":
    # Create converter
    converter = MicroscopeVolumeToPointCloud(
        target_points=2048,
        voxel_size=(1, 0.1, 0.1),  # z, y, x in micrometers
        intensity_threshold=None,  # Auto Otsu
    )

    # Option 1: Load from file
    # volume = converter.load_volume("microscope_stack.tif")

    # Option 2: Synthetic example (simulating cell-like structures)
    np.random.seed(42)
    volume = np.zeros((50, 128, 128))

    # Add some blob-like structures (simulating cells)
    for _ in range(5):
        z, y, x = (
            np.random.randint(10, 40),
            np.random.randint(20, 108),
            np.random.randint(20, 108),
        )
        radius = np.random.randint(8, 15)
        zz, yy, xx = np.ogrid[:50, :128, :128]
        mask = ((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) <= radius**2
        volume[mask] += np.random.uniform(0.5, 1.0)

    # Add noise
    volume += np.random.normal(0, 0.1, volume.shape)
    volume = np.clip(volume, 0, 1)

    # Convert to point cloud
    pointcloud = converter.convert(volume, normalize=True)

    print(f"Point cloud shape: {pointcloud['points'].shape}")
    print(
        f"Coordinates range: [{pointcloud['coords'].min():.3f}, {pointcloud['coords'].max():.3f}]"
    )
    print(f"Features shape: {pointcloud['features'].shape}")

    # Prepare for PointBERT
    points_tensor = prepare_for_pointbert(pointcloud, use_features=True)
    print(f"\nPointBERT input shape: {points_tensor.shape}")
    print(
        f"Ready for batch processing: add batch dimension -> {points_tensor.unsqueeze(0).shape}"
    )

    # For PointBERT model:
    # model = PointBERT(...)
    # batch = points_tensor.unsqueeze(0)  # Add batch dimension
    # features = model(batch)  # Extract features
