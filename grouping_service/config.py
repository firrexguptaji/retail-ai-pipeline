import os

# Clustering
 
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", 0.6))

CLUSTERING_METRIC = os.getenv("CLUSTERING_METRIC", "euclidean")
CLUSTERING_LINKAGE = os.getenv("CLUSTERING_LINKAGE", "average")

 
# Embedding
 
MODEL_NAME = os.getenv("MODEL_NAME", "resnet18")

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))

 
# Spatial weighting

SPATIAL_WEIGHT = float(os.getenv("SPATIAL_WEIGHT", 0.1))


# Device: "cpu" or "cuda" (falls back to CPU if CUDA is unavailable)
DEVICE = os.getenv("DEVICE", "cpu")

# Crops per embedding forward pass
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

#Output
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    os.path.join(BASE_DIR, "outputs")
)