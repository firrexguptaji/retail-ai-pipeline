
# 🛒 Retail AI Pipeline – Product Detection & Grouping

## 📌 Overview

This project implements a **production-style AI pipeline** to detect and group similar retail products from shelf images.

It is built using a **microservice architecture** with clear separation of responsibilities:

* **Detector Service** → detects product bounding boxes
* **Grouping Service** → groups visually + spatially similar products
* **Gateway Service** → orchestrates services and provides UI

---

## 🏗️ Architecture

```

Client (UI)
↓
Gateway Service (Flask)
↓
Detector Service (YOLO)
↓
Grouping Service (ResNet + Clustering)
↓
Final Output (Image + JSON)

````

---

## ⚙️ Services

### 1️⃣ Gateway Service

* Handles image upload
* Orchestrates detector → grouping
* Returns final JSON + image
* Serves output images

---

### 2️⃣ Detector Service

* Uses **YOLOv8**
* Implements:

  * Sliding window detection
  * Low confidence threshold
  * Area filtering
  * Non-Max Suppression (NMS)
  * Aspect ratio filtering
  * Duplicate removal
* Outputs:

  * Bounding boxes
  * Cropped images

---

### 3️⃣ Grouping Service

* Uses **ResNet18 embeddings**
* Combines:

  * Visual features
  * Spatial features
* Applies:

  * Feature normalization
  * Agglomerative clustering
* Outputs:

  * Group IDs
  * Annotated image

---

## 🧠 Key Design Decisions

* **Microservices** → modular & scalable
* **Sliding window detection** → improves recall
* **Spatial + visual fusion** → better grouping
* **Config-driven system** → no hardcoding
* **Logging** → easier debugging

---

## 🚀 How to Run
##  Dev Mode
### 🔹 1. Create Virtual Environment (Recommended)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
````

#### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 🔹 2. Install dependencies

```bash
pip install -r requirements/requirements-dev.txt
```

---

### 🔹 3. Start services

#### Option A (recommended)

```bash
python run_all.py
```

#### Option B (manual)

```bash
# Terminal 1
cd detector_service
python app.py

# Terminal 2
cd grouping_service
python app.py

# Terminal 3
cd gateway
python app.py
```

---

### 🔹 4. Open UI

```
http://127.0.0.1:5000
```

---

## 🐳 Docker Deployment

### 🔹 Run with Docker

```bash
cd docker
docker-compose up --build
```

---

### 🔹 Access UI

```
http://localhost:5000
```

---

### 🔹 Service Ports

| Service  | Port |
| -------- | ---- |
| Gateway  | 5000 |
| Detector | 8001 |
| Grouping | 8002 |

---

## 🧪 API Flow

1. Upload image → Gateway
2. Gateway → Detector
3. Gateway → Grouping
4. Final response returned

---

### Example Response

```json
{
  "request_id": "...",
  "output_image": "/outputs/result_xxx.jpg",
  "results": [
    {
      "bbox": [x1, y1, x2, y2],
      "group_id": 0
    }
  ]
}
```

---

### Error Responses

All services return errors as JSON. The `error` message is for people; `code` is stable for programs. Internal exception details are only written to the logs.

```json
{ "error": "Uploaded file is not a valid image", "code": "invalid_image" }
```

| Status | `code` | When |
|---|---|---|
| 400 | `missing_image` | No `image` field in the upload |
| 400 | `empty_file` / `invalid_image` | Empty, corrupt or non-image file |
| 400 | `invalid_payload` | Grouping received a malformed detector payload |
| 413 | `payload_too_large` | Upload larger than `MAX_UPLOAD_MB` (default 10) |
| 413 | `image_too_large` | Image larger than `MAX_IMAGE_PIXELS` (default 40 MP) |
| 415 | `unsupported_image_type` | Not JPEG, PNG, WEBP or BMP |
| 500 | `internal_error` | Unexpected failure; response includes an `error_id` to find in the logs |
| 502 | `upstream_error` | Detector or grouping service failed or is unreachable |
| 504 | `upstream_timeout` | Detector or grouping exceeded `REQUEST_TIMEOUT` (default 30 s) |

---

## ✅ Running Tests

```bash
pip install -r requirements/requirements-dev.txt
pytest                 # fast: models are stubbed, no weights needed
```

---

## 📂 Project Structure

```
project/
│
├── notebooks/
│   └── model.ipynb
├── gateway/
├── detector_service/
├── grouping_service/
├── logs/
├── docs/
├── models/
├── outputs/
├── docker/
│   └── docker-compose.yml
├── run_all.py
├── requirements.txt
└── README.md
```

---

## 📊 Notebook (Model Development)

The notebook (`notebooks/model.ipynb`) was used during the experimentation phase.

### 🔍 Purpose

* Prototyping detection pipeline
* Testing slicing strategy
* Developing filtering logic
* Validating grouping approach
* Analyzing clustering behavior

---

### 🔄 Transition to Production

| Notebook         | Production         |
| ---------------- | ------------------ |
| Inline code      | Microservices      |
| Hardcoded values | Config-driven      |
| Sequential flow  | API-based pipeline |

---

### 🎯 Key Learnings Applied

* Sliding window improves recall
* Area filtering removes noise
* Spatial + visual features improve grouping
* Normalization stabilizes clustering

---

## 🔧 Configuration & Tuning

All parameters are configurable via `config.py`.

---

### 🎯 Detector Config

```python
SLICE_SIZE = 512
OVERLAP = 0.4
IOU_THRESHOLD = 0.4
MIN_AREA_RATIO = 0.0005
MAX_AREA_RATIO = 0.03
```

---

### 🎯 Grouping Config

```python
DISTANCE_THRESHOLD = 0.6
SPATIAL_WEIGHT = 0.1
IMAGE_SIZE = 224
CLUSTERING_METRIC = "euclidean"
CLUSTERING_LINKAGE = "average"
```

---

### 🔍 Parameter Insights

* **DISTANCE_THRESHOLD**

  * lower → more groups
  * higher → fewer groups

* **SPATIAL_WEIGHT**

  * 0 → visual only
  * 0.1 → balanced
  * higher → spatial bias

---

### 🧪 Experiments

* Increase `DISTANCE_THRESHOLD` → merge clusters
* Reduce `SPATIAL_WEIGHT` → visual grouping
* Increase `OVERLAP` → better detection

---

### ⚙️ Environment Variables

```bash
DISTANCE_THRESHOLD=0.7
SPATIAL_WEIGHT=0.2
```

---

## 🖼️ Example Output

> Below is an example demonstrating detection and grouping results from the pipeline.

### 📥 Input Image

![Input Image](docs/input_example.jpg)

---

### 📤 Output (Grouped Products)

![Output Image](docs/output_example.jpg)

---

### 📊 Sample JSON Output

```json
{
  "request_id": "example-id",
  "output_image": "/outputs/result_example.jpg",
  "results": [
    {
      "bbox": [100, 200, 300, 400],
      "group_id": 2
    },
    {
      "bbox": [320, 210, 500, 390],
      "group_id": 2
    }
  ]
}
```

---

## ⚡ Features

* End-to-end ML pipeline
* Microservice architecture
* UI + API integration
* Config-driven design
* Logging support

---

## 🔮 Future Improvements

* Fine-tuned embeddings
* Better clustering (DBSCAN / metric learning)
* Independent service scaling

---

## 🏁 Conclusion

This project demonstrates the transition from:

```
Notebook → Production-ready ML system
```

Combining:

* Machine Learning
* Backend Engineering
* System Design

---

## 👤 Author

Aman Gupta

