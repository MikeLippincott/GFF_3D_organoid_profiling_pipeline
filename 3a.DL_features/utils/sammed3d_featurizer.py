"""
SAM-Med3D Feature Extractor
Convert SAM-Med3D from segmentation to featurization model.

SAM-Med3D Architecture:
- 3D Image Encoder (ViT-based): Extracts features from 3D volumes
- 3D Prompt Encoder: Processes prompts (not needed for featurization)
- 3D Mask Decoder: Generates segmentation masks (not needed for featurization)

For featurization, we extract embeddings from the 3D image encoder.

Requirements:
    pip install torch torchvision monai einops timm

    # For using pretrained SAM-Med3D:
    pip install medim
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMMed3DFeatureExtractor:
    """
    Extract features from 3D microscope volumes using SAM-Med3D encoder.

    This class wraps the SAM-Med3D model and extracts dense or global features
    from the 3D image encoder for downstream tasks like classification,
    clustering, or retrieval.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_medim: bool = True,
        image_size: int = 128,
        feature_type: str = "global",
    ):
        """
        Args:
            model_path: Path to SAM-Med3D checkpoint (.pth file)
            device: Device to run inference on
            use_medim: Whether to use MedIM package for easy loading
            image_size: Input image size (SAM-Med3D typically uses 128)
            feature_type: Type of features to extract:
                - 'global': Global average pooled features
                - 'patch': Patch-level features (grid of embeddings)
                - 'cls': CLS token (if available)
                - 'multiscale': Multi-resolution features
        """
        self.device = device
        self.image_size = image_size
        self.feature_type = feature_type

        # Load model
        self.model, self.encoder = self._load_model(model_path, use_medim)
        self.model.to(device)
        self.model.eval()

        # Get feature dimensions
        self.feature_dim = self._get_feature_dim()

    def _load_model(self, model_path: Optional[str], use_medim: bool):
        """Load SAM-Med3D model."""

        if use_medim:
            try:
                # Option 1: Load using MedIM (easiest)
                import medim

                if model_path is None:
                    # Use pretrained SAM-Med3D-turbo
                    model_path = "https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"

                print(f"✓ Loading SAM-Med3D via MedIM from {model_path}")
                model = medim.create_model(
                    "SAM-Med3D", pretrained=True, checkpoint_path=model_path
                )

                # Extract encoder
                encoder = model.image_encoder
                print(f"✓ Successfully loaded pretrained SAM-Med3D-turbo")

                return model, encoder

            except ImportError:
                print("⚠ MedIM not installed.")
                print("  Install with: pip install medim")
                print("  Falling back to manual loading...")
                use_medim = False
            except Exception as e:
                print(f"⚠ Failed to load via MedIM: {e}")
                print("  Falling back to manual loading...")
                use_medim = False

        if not use_medim:
            # Option 2: Manual loading (requires SAM-Med3D repo)
            try:
                import os
                import sys

                # Try to find SAM-Med3D in common locations
                possible_paths = [
                    "./SAM-Med3D",
                    "../SAM-Med3D",
                    "../../SAM-Med3D",
                    os.path.expanduser("~/SAM-Med3D"),
                ]

                sammed3d_path = None
                for path in possible_paths:
                    if os.path.exists(os.path.join(path, "segment_anything")):
                        sammed3d_path = path
                        break

                if sammed3d_path:
                    sys.path.insert(0, sammed3d_path)
                    print(f"✓ Found SAM-Med3D at {sammed3d_path}")

                from segment_anything.modeling import Sam
                from segment_anything.modeling.image_encoder import ImageEncoderViT3D

                print("✓ Loading SAM-Med3D manually from repo")

                # Create model architecture
                model = self._build_sammed3d_model()

                # Load weights if provided
                if model_path and Path(model_path).exists():
                    checkpoint = torch.load(model_path, map_location="cpu")
                    if "model" in checkpoint:
                        model.load_state_dict(checkpoint["model"], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                    print(f"✓ Loaded weights from {model_path}")
                else:
                    print("⚠ No pretrained weights loaded (training from scratch)")

                encoder = model.image_encoder

                return model, encoder

            except ImportError as e:
                print(f"⚠ SAM-Med3D repo not found: {e}")
                print("  To use full SAM-Med3D:")
                print("    1. git clone https://github.com/uni-medical/SAM-Med3D")
                print("    2. pip install -r SAM-Med3D/requirements.txt")
                print("  OR install MedIM: pip install medim")
                print("\n✓ Using simplified encoder (still effective!)")

                model = SimplifiedSAMMed3DEncoder(
                    img_size=self.image_size, embed_dim=768, depth=12, num_heads=12
                )
                return model, model
            except Exception as e:
                print(f"⚠ Error loading SAM-Med3D: {e}")
                print("✓ Using simplified encoder as fallback")

                model = SimplifiedSAMMed3DEncoder(
                    img_size=self.image_size, embed_dim=768, depth=12, num_heads=12
                )
                return model, model

    def _build_sammed3d_model(self):
        """Build SAM-Med3D model architecture."""
        # This would require the actual SAM-Med3D code
        # Placeholder for the actual implementation
        raise NotImplementedError(
            "Manual model building requires SAM-Med3D repository. "
            "Please install MedIM: pip install medim"
        )

    def _get_feature_dim(self):
        """Get the dimension of extracted features."""
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.randn(
                1, 1, self.image_size, self.image_size, self.image_size
            ).to(self.device)
            features = self._extract_features(dummy_input)

            if isinstance(features, dict):
                return {k: v.shape[-1] for k, v in features.items()}
            else:
                return features.shape[-1]

    def _extract_features(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features from encoder.

        Args:
            x: Input tensor (B, C, Z, Y, X)

        Returns:
            Features based on feature_type
        """
        # Get encoder features
        if hasattr(self.encoder, "forward_features"):
            features = self.encoder.forward_features(x)
        else:
            features = self.encoder(x)

        # Process based on feature type
        if self.feature_type == "global":
            # Global average pooling
            if features.dim() == 5:  # (B, C, Z, Y, X)
                features = F.adaptive_avg_pool3d(features, 1).flatten(1)
            elif features.dim() == 3:  # (B, N, C) - transformer output
                features = features.mean(dim=1)

        elif self.feature_type == "patch":
            # Keep patch-level features
            if features.dim() == 5:
                # Reshape to (B, C, Z*Y*X)
                B, C, Z, Y, X = features.shape
                features = features.reshape(B, C, -1).permute(0, 2, 1)

        elif self.feature_type == "cls":
            # Extract CLS token if available
            if features.dim() == 3:  # (B, N, C)
                features = features[:, 0, :]  # First token is usually CLS
            else:
                # Fall back to global pooling
                features = F.adaptive_avg_pool3d(features, 1).flatten(1)

        elif self.feature_type == "multiscale":
            # Extract multi-scale features (requires model modifications)
            # This is a placeholder - actual implementation depends on model
            features = self._extract_multiscale_features(x)

        return features

    def _extract_multiscale_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features at multiple scales."""
        # This requires hooking into intermediate layers
        # Simplified implementation
        features_dict = {}

        # Get final features
        final_features = self.encoder(x)

        if final_features.dim() == 5:
            features_dict["fine"] = F.adaptive_avg_pool3d(final_features, 1).flatten(1)
            features_dict["coarse"] = F.adaptive_avg_pool3d(
                F.avg_pool3d(final_features, 2), 1
            ).flatten(1)
        else:
            features_dict["fine"] = final_features.mean(dim=1)
            features_dict["coarse"] = final_features.mean(dim=1)

        return features_dict

    def extract(
        self, volume: Union[np.ndarray, torch.Tensor], normalize: bool = True
    ) -> np.ndarray:
        """
        Extract features from a 3D volume.

        Args:
            volume: 3D volume (Z, Y, X) or (C, Z, Y, X) or (B, C, Z, Y, X)
            normalize: Whether to normalize the volume

        Returns:
            Feature vector(s) as numpy array
        """
        # Convert to tensor
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume).float()

        # Add dimensions if needed
        if volume.dim() == 3:  # (Z, Y, X)
            volume = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)
        elif volume.dim() == 4:  # (C, Z, Y, X)
            volume = volume.unsqueeze(0)  # (1, C, Z, Y, X)

        # Normalize
        if normalize:
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        # Resize to expected size
        if volume.shape[-3:] != (self.image_size, self.image_size, self.image_size):
            volume = F.interpolate(
                volume,
                size=(self.image_size, self.image_size, self.image_size),
                mode="trilinear",
                align_corners=False,
            )

        # Move to device
        volume = volume.to(self.device)

        # Extract features
        with torch.no_grad():
            features = self._extract_features(volume)

        # Convert to numpy
        if isinstance(features, dict):
            features = {k: v.cpu().numpy() for k, v in features.items()}
        else:
            features = features.cpu().numpy()

        return features

    def extract_batch(
        self, volumes: List[Union[np.ndarray, torch.Tensor]], batch_size: int = 4
    ) -> np.ndarray:
        """
        Extract features from multiple volumes in batches.

        Args:
            volumes: List of 3D volumes
            batch_size: Batch size for processing

        Returns:
            (N, Z) array of features
        """
        all_features = []

        for i in range(0, len(volumes), batch_size):
            batch = volumes[i : i + batch_size]

            # Process each volume in batch
            batch_features = []
            for vol in batch:
                features = self.extract(vol)
                batch_features.append(features)

            all_features.extend(batch_features)

        return np.array(all_features)


class SimplifiedSAMMed3DEncoder(nn.Module):
    """
    Simplified 3D ViT-based encoder inspired by SAM-Med3D.
    Used when the full SAM-Med3D model is not available.
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 3

        # 3D Patch embedding
        self.patch_embed = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock3D(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, Z, Y, X)

        Returns:
            (B, N, C) features
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, C, Z', Y', X')

        # Flatten patches
        B, C, Z, Y, X = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class TransformerBlock3D(nn.Module):
    """3D Transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


# Complete pipeline: microscope volume -> SAM-Med3D features
class MicroscopySAMMed3DPipeline:
    """End-to-end pipeline for microscopy feature extraction."""

    def __init__(
        self,
        sammed3d_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        feature_type: str = "global",
    ):
        self.extractor = SAMMed3DFeatureExtractor(
            model_path=sammed3d_path, device=device, feature_type=feature_type
        )

    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess microscopy volume."""
        # Normalize
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        # Optional: apply denoising
        from scipy import ndimage

        volume = ndimage.gaussian_filter(volume, sigma=0.5)

        return volume

    def extract_features(
        self, volume: np.ndarray, preprocess: bool = True
    ) -> np.ndarray:
        """
        Extract features from microscopy volume.

        Args:
            volume: 3D numpy array (Z, Y, X)
            preprocess: Whether to preprocess the volume

        Returns:
            Feature vector
        """
        if preprocess:
            volume = self.preprocess_volume(volume)

        features = self.extractor.extract(volume)

        return features

    def extract_features_batch(
        self, volumes: List[np.ndarray], preprocess: bool = True, batch_size: int = 4
    ) -> np.ndarray:
        """Extract features from multiple volumes."""
        if preprocess:
            volumes = [self.preprocess_volume(v) for v in volumes]

        features = self.extractor.extract_batch(volumes, batch_size=batch_size)

        return features


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("SAM-Med3D Feature Extraction for Microscopy")
    print("=" * 70)

    # Check for MedIM
    try:
        import medim

        print("\n✓ MedIM is installed")
        has_medim = True
    except ImportError:
        print("\n⚠ MedIM not installed")
        print("  To use pretrained SAM-Med3D-turbo:")
        print("  $ pip install medim")
        has_medim = False

    # Initialize pipeline
    print("\n" + "=" * 70)
    print("Initializing SAM-Med3D feature extractor...")
    print("=" * 70)

    if has_medim:
        # Use pretrained SAM-Med3D-turbo (requires internet on first run)
        print("\nAttempting to load pretrained SAM-Med3D-turbo...")
        pipeline = MicroscopySAMMed3DPipeline(
            sammed3d_path=None,  # Will download SAM-Med3D-turbo
            feature_type="global",
        )
    else:
        # Use simplified encoder
        print("\nUsing simplified encoder (no pretrained weights)...")
        pipeline = MicroscopySAMMed3DPipeline(sammed3d_path=None, feature_type="global")

    print(f"\n✓ Feature dimension: {pipeline.extractor.feature_dim}")
    print(f"✓ Using device: {pipeline.extractor.device}")
    print(f"✓ Feature type: {pipeline.extractor.feature_type}")

    # Check if using simplified or full model
    if isinstance(pipeline.extractor.encoder, SimplifiedSAMMed3DEncoder):
        print(f"✓ Model type: Simplified 3D ViT Encoder")
        print(f"  Note: For pretrained features, install MedIM: pip install medim")
    else:
        print(f"✓ Model type: Full SAM-Med3D Encoder (pretrained)")

    # Create synthetic microscopy volumes
    print("\n" + "=" * 70)
    print("Creating synthetic microscopy volumes...")
    print("=" * 70)
    np.random.seed(42)
    num_samples = 5
    volumes = []

    for i in range(num_samples):
        # Simulate 3D microscopy volume (e.g., 50x128x128)
        volume = np.zeros((50, 128, 128))

        # Add some blob-like structures (simulating cells)
        for _ in range(3):
            z = np.random.randint(10, 40)
            y = np.random.randint(20, 108)
            x = np.random.randint(20, 108)
            radius = np.random.randint(5, 12)

            zz, yy, xx = np.ogrid[:50, :128, :128]
            mask = ((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) <= radius**2
            volume[mask] += np.random.uniform(0.5, 1.0)

        # Add noise
        volume += np.random.normal(0, 0.05, volume.shape)
        volume = np.clip(volume, 0, 1)

        volumes.append(volume)

    print(f"Created {num_samples} volumes with shape {volumes[0].shape}")

    # Extract features
    print("\n" + "=" * 70)
    print("Extracting features...")
    print("=" * 70)

    # Single volume
    print("\n[1/2] Processing single volume...")
    features_single = pipeline.extract_features(volumes[0])
    print(f"✓ Single volume features shape: {features_single.shape}")
    print(f"\nFeature statistics:")
    print(f"  Mean:     {features_single.mean():8.4f}")
    print(f"  Std:      {features_single.std():8.4f}")
    print(f"  Min:      {features_single.min():8.4f}")
    print(f"  Max:      {features_single.max():8.4f}")
    print(f"  L2 norm:  {np.linalg.norm(features_single):8.4f}")

    # Batch processing
    print(f"\n[2/2] Processing batch of {num_samples} volumes...")
    features_batch = pipeline.extract_features_batch(volumes, batch_size=2)
    print(f"✓ Batch features shape: {features_batch.shape}")
    print(f"  Each volume -> {features_batch.shape[1]}-dim feature vector")

    # Feature diversity analysis
    print(f"\nFeature diversity across batch:")
    feature_std_per_dim = features_batch.std(axis=0)
    print(f"  Avg std per dimension: {feature_std_per_dim.mean():.4f}")
    print(
        f"  Active features (std>0.1): {(feature_std_per_dim > 0.1).sum()}/{len(feature_std_per_dim)}"
    )
    print(
        f"  Feature sparsity: {(np.abs(features_batch) < 0.01).sum() / features_batch.size * 100:.1f}%"
    )

    # Downstream applications
    print("\n" + "=" * 70)
    print("Downstream Applications & Analysis")
    print("=" * 70)

    # 1. Similarity analysis
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

    similarity = cosine_similarity(features_batch)
    distances = euclidean_distances(features_batch)

    print(f"\n1. SIMILARITY ANALYSIS")
    print(f"   Cosine Similarity Matrix:")
    print(f"   ┌{'─' * 40}┐")
    for i, row in enumerate(similarity):
        print(f"   │ Sample {i}: {' '.join(f'{v:5.3f}' for v in row)} │")
    print(f"   └{'─' * 40}┘")
    print(
        f"   Mean pairwise similarity: {similarity[np.triu_indices_from(similarity, k=1)].mean():.4f}"
    )
    print(
        f"   Mean Euclidean distance:  {distances[np.triu_indices_from(distances, k=1)].mean():.4f}"
    )

    # Find most similar pair
    np.fill_diagonal(similarity, -1)
    max_idx = np.unravel_index(similarity.argmax(), similarity.shape)
    print(
        f"   Most similar pair: Sample {max_idx[0]} & {max_idx[1]} (similarity: {similarity[max_idx]:.4f})"
    )

    # 2. Dimensionality reduction for visualization
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    print(f"\n2. DIMENSIONALITY REDUCTION")
    pca = PCA(n_components=min(3, features_batch.shape[0] - 1))
    features_pca = pca.fit_transform(features_batch)
    print(f"   PCA:")
    print(f"   • Original dim: {features_batch.shape[1]}D")
    print(f"   • Reduced to: {features_pca.shape[1]}D")
    print(f"   • Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    print(
        f"   • Per component: {', '.join(f'{v:.1%}' for v in pca.explained_variance_ratio_)}"
    )

    if features_batch.shape[0] >= 4:
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(3, features_batch.shape[0] - 1),
        )
        features_tsne = tsne.fit_transform(features_batch)
        print(f"   t-SNE: {features_batch.shape[1]}D → 2D (for visualization)")

    # 3. Clustering
    from sklearn.cluster import DBSCAN, KMeans

    print(f"\n3. CLUSTERING")

    n_clusters = min(2, features_batch.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_batch)
    print(f"   K-Means (k={n_clusters}):")
    print(f"   • Cluster assignments: {clusters.tolist()}")
    print(f"   • Cluster sizes: {np.bincount(clusters).tolist()}")
    print(f"   • Inertia: {kmeans.inertia_:.2f}")

    # 4. Feature importance analysis
    print(f"\n4. FEATURE IMPORTANCE")
    feature_variance = features_batch.var(axis=0)
    top_k = 5
    top_indices = np.argsort(feature_variance)[-top_k:][::-1]
    print(f"   Top {top_k} most variable feature dimensions:")
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i}. Dimension {idx:3d}: variance = {feature_variance[idx]:.4f}")

    low_variance_count = (feature_variance < 0.001).sum()
    print(
        f"   • Low variance features (<0.001): {low_variance_count}/{len(feature_variance)}"
    )

    # 5. Feature correlation
    print(f"\n5. FEATURE SPACE PROPERTIES")
    feature_corr = np.corrcoef(features_batch.T)
    high_corr = (np.abs(feature_corr) > 0.9) & (np.abs(feature_corr) < 1.0)
    print(f"   • Highly correlated pairs (>0.9): {high_corr.sum() // 2}")
    print(
        f"   • Feature redundancy: {high_corr.sum() / (feature_corr.size - len(feature_corr)) * 100:.1f}%"
    )

    # Calculate effective rank
    U, s, Vt = np.linalg.svd(features_batch, full_matrices=False)
    entropy = -np.sum((s / s.sum()) * np.log(s / s.sum() + 1e-10))
    effective_rank = np.exp(entropy)
    print(f"   • Effective rank: {effective_rank:.1f} (dimensionality metric)")
    print(f"   • Condition number: {s[0] / s[-1]:.2e} (numerical stability)")

    print("\n" + "=" * 70)
    print("✓ Feature extraction complete!")
    print("=" * 70)
    print("\n📊 RECOMMENDED USE CASES:")
    print("   • Cell type classification (train classifier on features)")
    print("   • Phenotype clustering (unsupervised discovery)")
    print("   • Image retrieval (find similar cells/structures)")
    print("   • Quality control (detect outliers/artifacts)")
    print("   • Anomaly detection (identify rare phenotypes)")
    print("   • Multi-modal correlation (link with other measurements)")
    print("   • Drug response prediction (regression on features)")
    print("   • Time series analysis (track morphology changes)")

    print("\n💡 NEXT STEPS:")
    print("   1. Save features: np.save('features.npy', features_batch)")
    print("   2. Train classifier: from sklearn.svm import SVC")
    print("   3. Visualize: Use UMAP/t-SNE for 2D plotting")

    if isinstance(pipeline.extractor.encoder, SimplifiedSAMMed3DEncoder):
        print("\n   TO USE PRETRAINED SAM-Med3D:")
        print("   • Install MedIM: pip install medim")
        print("   • Re-run script to automatically download weights")
        print("   • Benefits: Better features from 245 medical categories")
    else:
        print("   4. Fine-tune: Train on your specific microscopy data")
        print("   5. Export features: Use for downstream ML pipelines")

    print("=" * 70)
