import cv2
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from torchvision import models, transforms

from common.logger import setup_logger

from .config import (
    CLUSTERING_LINKAGE,
    CLUSTERING_METRIC,
    DEVICE,
    DISTANCE_THRESHOLD,
    EMBED_BATCH_SIZE,
    IMAGE_SIZE,
    MODEL_NAME,
    SPATIAL_WEIGHT,
)

logger = setup_logger("grouping")

# ImageNet statistics the torchvision ResNet weights were trained with.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBEDDING_DIM = 512  # ResNet18 penultimate layer


# -------------------------------
# Device and model (loaded once, at service start)
# -------------------------------
def resolve_device(requested: str) -> torch.device:
    """Use the requested device, falling back to CPU if CUDA is not available."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("DEVICE=cuda requested but CUDA is not available; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


device = resolve_device(DEVICE)
_model = None


def load_model():
    """Build the embedding model once and keep it on `device`."""
    global _model
    if _model is not None:
        return _model

    if MODEL_NAME != "resnet18":
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    # IMAGENET1K_V1 = the same weights the deprecated `pretrained=True` loaded.
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(backbone.children())[:-1])  # drop the classifier
    model.eval()
    _model = model.to(device)

    logger.info(f"Loaded model: {MODEL_NAME} on {device}")
    return _model


# -------------------------------
# Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess(crops) -> torch.Tensor:
    """BGR uint8 crops (OpenCV) -> normalised float tensor of shape (N, 3, IMAGE_SIZE, IMAGE_SIZE)."""
    return torch.stack([transform(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops])


# -------------------------------
# Embedding Extraction
# -------------------------------
def get_embeddings(crops, batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """Return an (N, 512) array, running the model once per batch."""
    if len(crops) == 0:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = load_model()
    batches = []

    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = preprocess(crops[start:start + batch_size]).to(device)
            batches.append(model(batch).flatten(1).cpu().numpy())

    embeddings = np.concatenate(batches)
    logger.info(f"Computed embeddings for {len(crops)} crops")
    return embeddings


# -------------------------------
# Grouping Logic
# -------------------------------
def group_products(crops, boxes, image_shape):
    """Return one integer group label per crop."""
    if len(crops) != len(boxes):
        raise ValueError("crops and boxes must have the same length")

    # Agglomerative clustering needs at least 2 samples.
    if len(crops) == 0:
        return []
    if len(crops) == 1:
        logger.info("Single detection: assigned to group 0 without clustering")
        return [0]

    embeddings = get_embeddings(crops)

    h, w = image_shape[:2]

    # Spatial features: normalised box centre (notebook logic)
    centres = np.array([[(x1 + x2) / 2 / w, (y1 + y2) / 2 / h] for x1, y1, x2, y2 in boxes])
    final_embeddings = np.concatenate([embeddings, SPATIAL_WEIGHT * centres], axis=1)

    # Normalize embeddings
    norms = np.linalg.norm(final_embeddings, axis=1, keepdims=True)
    final_embeddings = final_embeddings / (norms + 1e-6)

    # Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        metric=CLUSTERING_METRIC,
        linkage=CLUSTERING_LINKAGE
    )

    labels = clustering.fit_predict(final_embeddings)

    logger.info(
        f"Clustering produced {len(set(labels))} groups for {len(crops)} items"
    )

    return labels.tolist()
