# Contour Lab

Contour Lab is a full-stack image-processing application built around the custom
algorithms in [`notebooks/CannyEdge_LineDetection.ipynb`](./notebooks/CannyEdge_LineDetection.ipynb).
It keeps the original image matrix in a backend session and exposes the
Gaussian blur, Sobel gradient, non-maximum suppression, hysteresis, Hough
accumulator, and detected-line stages to a React UI.

## Architecture

- `backend/core/` contains the notebook-derived, NumPy implementations.
- `backend/models/` contains Pydantic contracts and the thread-safe
  `ImageManager`, which caches original and intermediate matrices in memory.
- `backend/controllers/` runs Canny and Hough processing.
- `backend/views/` contains FastAPI routes.
- `frontend/` is a Vite + React client with parameter controls, output routing,
  and step-by-step image viewers.

The only OpenCV operations are image decoding/encoding and line rendering; the
Canny and Hough computations themselves are implemented from scratch.

## Run locally

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
uvicorn backend.main:app --reload
```

The API starts at `http://localhost:8000`. Interactive API documentation is
available at `http://localhost:8000/docs`.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`), upload an image, adjust
the controls, and run the pipeline. Hough can consume either the custom Canny
edge matrix or a grayscale version of the original image.

## API

- `POST /api/images` — multipart image upload; returns an in-memory `image_id`.
- `GET /api/images/{image_id}` — returns the original image.
- `POST /api/images/{image_id}/canny` — runs Canny and returns all intermediate
  stages.
- `POST /api/images/{image_id}/hough` — runs Hough on the latest Canny output.
- `POST /api/images/{image_id}/pipeline` — runs both stages and routes the
  selected Hough input.

Example pipeline body:

```json
{
  "canny": {
    "kernel_size": 7,
    "sigma": 1,
    "low_threshold": 80,
    "high_threshold": 200
  },
  "hough": {
    "radius_res": 1,
    "theta_res": 1,
    "max_lines": 10,
    "vote_threshold": 20
  },
  "hough_input": "canny"
}
```
