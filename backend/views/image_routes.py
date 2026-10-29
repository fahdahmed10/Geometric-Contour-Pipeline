from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.controllers.processing import run_canny, run_hough
from backend.core.image_codec import decode_image, matrix_to_data_url
from backend.models.image_manager import image_manager
from backend.models.schemas import CannyParameters, HoughParameters, ImageResponse, PipelineRequest, ProcessResponse

router = APIRouter(tags=["image-processing"])


@router.post("/images", response_model=ImageResponse)
async def upload_image(file: UploadFile = File(...)) -> ImageResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a valid image file")
    try:
        image = decode_image(await file.read())
        image_id = image_manager.create(image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ImageResponse(image_id=image_id, width=image.shape[1], height=image.shape[0], channels=image.shape[2])


def _session_or_404(image_id: str):
    try:
        return image_manager.get(image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/images/{image_id}")
def get_image(image_id: str) -> dict[str, object]:
    session = _session_or_404(image_id)
    return {"image_id": image_id, "original": matrix_to_data_url(session.original, color=True)}


@router.post("/images/{image_id}/canny", response_model=ProcessResponse)
def process_canny(image_id: str, params: CannyParameters) -> ProcessResponse:
    _session_or_404(image_id)
    try:
        result = run_canny(image_manager, image_id, params)
        image_manager.cache(image_id, "latest_canny", image_manager.get(image_id).cache[f"canny:{params.model_dump_json()}"]["edges"])
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ProcessResponse(image_id=image_id, output=result["output"], steps=result["steps"], parameters=result["parameters"])


@router.post("/images/{image_id}/hough")
def process_hough(image_id: str, params: HoughParameters) -> dict[str, object]:
    session = _session_or_404(image_id)
    edges = session.cache.get("latest_canny")
    if edges is None:
        raise HTTPException(status_code=409, detail="Run Canny first or use the pipeline endpoint")
    try:
        result = run_hough(image_manager, edges, params)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"image_id": image_id, **result}


@router.post("/images/{image_id}/pipeline")
def process_pipeline(image_id: str, request: PipelineRequest) -> dict[str, object]:
    _session_or_404(image_id)
    try:
        canny_result = run_canny(image_manager, image_id, request.canny)
        canny_steps = canny_result["steps"]
        canny_edges = image_manager.get(image_id).cache[f"canny:{request.canny.model_dump_json()}"]["edges"]
        image_manager.cache(image_id, "latest_canny", canny_edges)
        hough_input = canny_edges
        if request.hough_input == "original":
            original = image_manager.get(image_id).original
            hough_input = (original[..., 0] * 0.299 + original[..., 1] * 0.587 + original[..., 2] * 0.114)
        hough_result = run_hough(image_manager, hough_input, request.hough)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"image_id": image_id, "original": matrix_to_data_url(image_manager.get(image_id).original, color=True), "canny": canny_result, "hough": hough_result}
