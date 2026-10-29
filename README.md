# Contour Lab

Contour Lab is a full-stack image-processing application built around the custom
algorithms in [`notebooks/CannyEdge_LineDetection.ipynb`](./notebooks/CannyEdge_LineDetection.ipynb).
It keeps the original image matrix in a backend session and exposes the
Gaussian blur, Sobel gradient, non-maximum suppression, hysteresis, Hough
accumulator, and detected-line stages to a React UI.

The frontend is organized as a guided two-stage workspace:

1. **Canny Edge Detection** starts from the original uploaded image, exposes all
   Canny parameters, and lets the user save the result to **Output 1** or
   **Output 2**.
2. **Hough Line Detection** accepts the original image, either Canny output slot,
   or the latest Canny result. Its rendered line overlay can independently be
   saved to **Output 1** or **Output 2**.

Each section displays a large `INPUT | Output 1 | Output 2` viewer grid followed
by its own intermediate-stage pipeline.

## Architecture

- `backend/core/` contains the notebook-derived, NumPy implementations.
- `backend/models/` contains Pydantic contracts and the thread-safe
  `ImageManager`, which caches original and intermediate matrices in memory.
- `backend/controllers/` runs Canny and Hough processing.
- `backend/views/` contains FastAPI routes.
- `frontend/` is a Vite + React client with parameter controls, output routing,
  separate Canny/Hough sections, large input/output viewers, and step-by-step
  image viewers.

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

Open the Vite URL (normally `http://localhost:5173`) and follow this workflow:

1. Upload an image.
2. Configure and run Canny.
3. Choose whether the Canny result is stored in Output 1 or Output 2.
4. Choose the Hough input from Original, Canny Output 1, Canny Output 2, or
   Latest Canny edges.
5. Configure and run Hough.
6. Choose the Hough destination output slot and inspect its line overlay.

## API

- `POST /api/images` — multipart image upload; returns an in-memory `image_id`.
- `GET /api/images/{image_id}` — returns the original image.
- `POST /api/images/{image_id}/canny` — runs Canny, returns all intermediate
  stages, and accepts `output_slot: "1" | "2"` to cache its matrix.
- `POST /api/images/{image_id}/hough` — runs Hough on `original`, `canny`,
  `output1`, or `output2`, accepts `output_slot: "1" | "2"`, and returns the
  edge input, accumulator, and rendered line overlay stages.
- `POST /api/images/{image_id}/pipeline` — runs both stages and routes the
  selected Hough input.

### Canny request

```json
{
  "kernel_size": 7,
  "sigma": 1,
  "low_threshold": 80,
  "high_threshold": 200,
  "output_slot": "1"
}
```

### Hough request

```json
{
  "radius_res": 1,
  "theta_res": 1,
  "max_lines": 10,
  "vote_threshold": 20,
  "input_source": "output1",
  "output_slot": "2"
}
```

`input_source` accepts `original`, `canny`, `output1`, or `output2`.

### Combined pipeline request

The combined endpoint remains available for clients that want to execute both
algorithms in one request:

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

## Intermediate stages

The Canny response includes:

- Grayscale
- Gaussian blur
- Gradient magnitude
- Gradient direction
- Non-maximum suppression
- Final hysteresis edges

The Hough response includes:

- Edge input
- Hough accumulator space
- Final detected-line overlay

All stages are returned as PNG data URLs with matrix shape and min/max metadata,
so the React client can render them without reloading the uploaded image.
