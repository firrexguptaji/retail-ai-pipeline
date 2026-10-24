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
    IMAGE_SIZE,
    MODEL_NAME,
    SPATIAL_WEIGHT,
)

logger = setup_logger("grouping")


  
# Load Model (Config-driven)
  
def load_model():
    if MODEL_NAME == "resnet18":
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    return model


resnet = load_model()
logger.info(f"Loaded model: {MODEL_NAME}")


  
# Transform
  
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


  
# Embedding Extraction
  
def get_embeddings(crops):
    embeddings = []

    with torch.no_grad():
        for crop in crops:
            # 🔥 BGR → RGB (IMPORTANT)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            inp = transform(crop).unsqueeze(0)

            if DEVICE == "cuda":
                inp = inp.cuda()
                resnet.cuda()

            emb = resnet(inp).squeeze().cpu().numpy()
            embeddings.append(emb)

    embeddings = np.array(embeddings)

    logger.info(f"Computed embeddings for {len(crops)} crops")

    return embeddings


  
# Grouping Logic
  
def group_products(crops, boxes, image_shape):
    if len(crops) == 0:
        return []

    embeddings = get_embeddings(crops)

    h, w = image_shape[:2]

    final_embeddings = []

    for emb, (x1, y1, x2, y2) in zip(embeddings, boxes):
          
        # Spatial Features (Notebook logic)
          
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        spatial = np.array([
            cx / w,
            cy / h
        ])

        combined = np.concatenate([
            emb,
            SPATIAL_WEIGHT * spatial
        ])

        final_embeddings.append(combined)

    final_embeddings = np.array(final_embeddings)

      
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