from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.controllers.processing import run_canny, run_canny_comparison, run_ellipse, run_hough
from backend.core.image_codec import decode_image, matrix_to_data_url
from backend.models.image_manager import image_manager
from backend.models.schemas import CannyComparisonRequest, CannyParameters, CannyRequest, EllipseParameters, EllipseRequest, HoughParameters, HoughRequest, ImageResponse, PipelineRequest, ProcessResponse

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
def process_canny(image_id: str, params: CannyRequest) -> ProcessResponse:
    _session_or_404(image_id)
    try:
        result = run_canny(image_manager, image_id, CannyParameters(**params.model_dump(exclude={"output_slot"})))
        edges = image_manager.get(image_id).cache[f"canny:{CannyParameters(**params.model_dump(exclude={'output_slot'})).model_dump_json()}"]["edges"]
        image_manager.cache(image_id, "latest_canny", edges)
        image_manager.cache(image_id, f"canny_output_{params.output_slot}", edges)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ProcessResponse(
        image_id=image_id,
        output=result["output"],
        opencv_output=result["opencv_output"],
        opencv_steps=result["opencv_steps"],
        steps=result["steps"],
        parameters=result["parameters"],
    )


@router.post("/images/{image_id}/canny-comparison")
def process_canny_comparison(image_id: str, params: CannyComparisonRequest) -> dict[str, object]:
    session = _session_or_404(image_id)
    try:
        return {"image_id": image_id, **run_canny_comparison(session.original, params)}
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/images/{image_id}/hough")
def process_hough(image_id: str, params: HoughRequest) -> dict[str, object]:
    session = _session_or_404(image_id)
    edges = session.cache.get("latest_canny")
    if params.input_source == "original":
        original = session.original
        edges = original[..., 0] * 0.299 + original[..., 1] * 0.587 + original[..., 2] * 0.114
    elif params.input_source.startswith("output"):
        edges = session.cache.get(f"canny_output_{params.input_source[-1]}")
    if edges is None:
        raise HTTPException(status_code=409, detail="Run Canny first or choose a populated Hough input")
    try:
        result = run_hough(image_manager, edges, HoughParameters(**params.model_dump(exclude={"input_source", "output_slot"})))
        image_manager.cache(image_id, f"hough_output_{params.output_slot}", result.pop("_output_matrix"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"image_id": image_id, **result}


@router.post("/images/{image_id}/ellipse")
def process_ellipse(image_id: str, params: EllipseRequest) -> dict[str, object]:
    session = _session_or_404(image_id)
    if params.input_source == "canny":
        edges = session.cache.get("latest_canny")
    else:
        edges = session.cache.get(f"canny_output_{params.input_source[-1]}")
    if edges is None:
        raise HTTPException(status_code=409, detail="Run Canny first or choose a populated Canny input")
    try:
        result = run_ellipse(
            session.original,
            edges,
            EllipseParameters(**params.model_dump(exclude={"input_source", "output_slot", "compare_opencv", "resize_input", "canny"})),
            compare_opencv=params.compare_opencv,
            canny_params=params.canny if params.input_source == "canny" else None,
            resize_input=params.resize_input,
        )
        image_manager.cache(image_id, f"ellipse_output_{params.output_slot}", result.pop("_output_matrix"))
    except (ValueError, TypeError) as exc:
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
