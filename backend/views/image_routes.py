from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.controllers.processing import run_canny, run_canny_comparison, run_circle, run_contour_metrics, run_ellipse, run_hough, run_snake
from backend.core.image_codec import decode_image, matrix_to_data_url
from backend.models.image_manager import image_manager
from backend.models.schemas import CannyComparisonRequest, CannyParameters, CannyRequest, CircleRequest, ContourMetricsRequest, EllipseParameters, EllipseRequest, HoughParameters, HoughRequest, ImageResponse, PipelineRequest, ProcessResponse
from backend.models.snake_models import SnakeRequest

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
        canny_params = CannyParameters(**params.model_dump(exclude={"output_slot"}))
        result = run_canny(image_manager, image_id, canny_params)
        cached_steps = image_manager.get(image_id).cache[f"canny:{canny_params.model_dump_json()}"]
        edges = cached_steps["edges"]
        image_manager.cache(image_id, "latest_canny", edges)
        image_manager.cache(image_id, f"canny_output_{params.output_slot}", edges)
        # Circle CHT needs the gradient direction from the same Gaussian-blurred
        # Canny input. Keep it beside the existing edge-slot cache; the public
        # edge-slot behavior remains unchanged.
        image_manager.cache(image_id, "latest_canny_blur", cached_steps["gaussian_blur"])
        image_manager.cache(image_id, f"canny_blur_output_{params.output_slot}", cached_steps["gaussian_blur"])
        image_manager.cache(image_id, "latest_canny_params", canny_params.model_dump())
        image_manager.cache(image_id, f"canny_params_output_{params.output_slot}", canny_params.model_dump())
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
        result = run_hough(
            image_manager,
            edges,
            HoughParameters(**params.model_dump(exclude={"input_source", "output_slot"})),
            original=session.original,
        )
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


@router.post("/images/{image_id}/circle")
def process_circle(image_id: str, params: CircleRequest) -> dict[str, object]:
    session = _session_or_404(image_id)
    if params.input_source == "canny":
        edges = session.cache.get("latest_canny")
        blurred = session.cache.get("latest_canny_blur")
        canny_data = session.cache.get("latest_canny_params")
    else:
        slot = params.input_source[-1]
        edges = session.cache.get(f"canny_output_{slot}")
        blurred = session.cache.get(f"canny_blur_output_{slot}")
        canny_data = session.cache.get(f"canny_params_output_{slot}")
    if edges is None or blurred is None:
        raise HTTPException(status_code=409, detail="Run Canny first or choose a populated Canny input")
    try:
        canny_params = CannyParameters(**canny_data) if isinstance(canny_data, dict) else None
        result = run_circle(
            session.original,
            edges,
            blurred,
            params,
            compare_opencv=params.compare_opencv,
            canny_params=canny_params,
        )
        image_manager.cache(image_id, f"circle_output_{params.output_slot}", result.pop("_output_matrix"))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"image_id": image_id, **result}


@router.post("/images/{image_id}/snake")
def process_snake(image_id: str, params: SnakeRequest) -> dict[str, object]:
    _session_or_404(image_id)
    try:
        result = run_snake(image_manager, image_id, params)
        image_manager.cache(image_id, f"snake_output_{params.output_slot}", result["output"])
        image_manager.cache(image_id, f"snake_contour_{params.output_slot}", result["final_contour"])
        image_manager.cache(image_id, "latest_snake_contour", result["final_contour"])
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"image_id": image_id, **result}


@router.post("/images/{image_id}/contour-metrics")
def process_contour_metrics(image_id: str, params: ContourMetricsRequest) -> dict[str, object]:
    session = _session_or_404(image_id)
    cached_key = {
        "snake": "latest_snake_contour",
        "output1": "snake_contour_1",
        "output2": "snake_contour_2",
    }.get(params.input_source)
    cached = session.cache.get(cached_key) if cached_key else None
    if params.input_source == "raw":
        if params.points is None:
            raise HTTPException(status_code=422, detail="Raw contour input requires points")
        points = [(point.x, point.y) for point in params.points]
    elif isinstance(cached, list):
        try:
            points = [(float(point["x"]), float(point["y"])) for point in cached]
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Cached contour has an invalid shape") from exc
    else:
        raise HTTPException(status_code=409, detail="Run Snake first or choose a populated contour input")
    try:
        result = run_contour_metrics(image_manager, image_id, points)
        image_manager.cache(image_id, f"contour_metrics_output_{params.output_slot}", result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"image_id": image_id, **result}


@router.post("/images/{image_id}/pipeline")
def process_pipeline(image_id: str, request: PipelineRequest) -> dict[str, object]:
    _session_or_404(image_id)
    try:
        canny_result = run_canny(image_manager, image_id, request.canny)
        canny_steps = canny_result["steps"]
        cached_canny = image_manager.get(image_id).cache[f"canny:{request.canny.model_dump_json()}"]
        canny_edges = cached_canny["edges"]
        image_manager.cache(image_id, "latest_canny", canny_edges)
        image_manager.cache(image_id, "latest_canny_blur", cached_canny["gaussian_blur"])
        image_manager.cache(image_id, "latest_canny_params", request.canny.model_dump())
        hough_input = canny_edges
        if request.hough_input == "original":
            original = image_manager.get(image_id).original
            hough_input = (original[..., 0] * 0.299 + original[..., 1] * 0.587 + original[..., 2] * 0.114)
        hough_result = run_hough(image_manager, hough_input, request.hough, original=image_manager.get(image_id).original)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"image_id": image_id, "original": matrix_to_data_url(image_manager.get(image_id).original, color=True), "canny": canny_result, "hough": hough_result}
