from typing import Any

from pydantic import BaseModel, Field, model_validator
from backend.models.snake_models import SnakePoint


class CannyParameters(BaseModel):
    kernel_size: int = Field(7, description="Odd kernel size supported by the notebook")
    sigma: float = Field(1.0, gt=0)
    low_threshold: float = Field(80.0, ge=0)
    high_threshold: float = Field(200.0, ge=0)


class CannyRequest(CannyParameters):
    output_slot: str = Field("1", pattern="^(1|2)$")


class HoughParameters(BaseModel):
    radius_res: float = Field(1.0, gt=0)
    theta_res: float = Field(1.0, gt=0, le=180)
    max_lines: int = Field(10, ge=1, le=100)
    vote_threshold: int = Field(20, ge=0)


class HoughRequest(HoughParameters):
    input_source: str = Field("canny", pattern="^(original|canny|output1|output2)$")
    output_slot: str = Field("2", pattern="^(1|2)$")


class EllipseParameters(BaseModel):
    a_min: float = Field(10.0, gt=0, description="Minimum allowed minor semi-axis in pixels")
    a_max: float = Field(150.0, gt=0, description="Maximum allowed major semi-axis in pixels")
    vote_threshold: int = Field(40, ge=10, le=500, description="Minimum supporting edge pixels")
    fit_tol: float = Field(2.0, ge=0.5, le=8.0, description="Maximum inlier distance in pixels")
    min_support: float = Field(0.5, ge=0.1, le=1.0, description="Minimum perimeter support fraction")
    max_ellipses: int = Field(3, ge=1, le=8, description="Maximum ellipses per component")
    ransac_iters: int = Field(300, ge=50, le=1500, description="RANSAC hypotheses per search")

    @model_validator(mode="after")
    def validate_axis_range(self):
        if self.a_min > self.a_max:
            raise ValueError("a_min must be less than or equal to a_max")
        return self


class EllipseRequest(EllipseParameters):
    input_source: str = Field("canny", pattern="^(canny|output1|output2)$")
    output_slot: str = Field("2", pattern="^(1|2)$")
    compare_opencv: bool = False
    resize_input: bool = True
    canny: CannyParameters = Field(default_factory=CannyParameters)


class CircleRequest(BaseModel):
    """Request contract for either independent notebook circle detector."""

    algorithm: str = Field("cht", pattern="^(cht|ransac)$")
    input_source: str = Field("canny", pattern="^(canny|output1|output2)$")
    output_slot: str = Field("2", pattern="^(1|2)$")
    compare_opencv: bool = False

    # Circular Hough Transform controls.
    min_diam: int = Field(45, ge=4, le=10000)
    max_diam: int = Field(125, ge=4, le=10000)
    min_votes: int = Field(48, ge=5, le=100000)
    min_coverage: float = Field(0.6, ge=0.1, le=1.0)
    perim_frac: float = Field(0.4, ge=0.0, le=1.0)
    smooth: int = Field(3, ge=1, le=9)
    nms_win: int = Field(7, ge=3, le=21)
    align_cos: float = Field(0.7, ge=0.0, le=0.95)
    refine: bool = True

    # RANSAC controls.
    tol: float = Field(2.0, ge=0.5, le=6.0)
    max_iter: int = Field(1000, ge=100, le=10000)
    min_inliers: int = Field(48, ge=5, le=100000)
    seed: int = Field(0, ge=0, le=99)

    # Optional OpenCV comparison controls. None keeps the notebook-derived
    # value (based on the selected custom detector and Canny settings).
    opencv_dp: float = Field(1.0, ge=0.5, le=3.0)
    opencv_min_dist: float | None = Field(None, ge=1.0, le=10000.0)
    opencv_param1: float | None = Field(None, ge=1.0, le=1000.0)
    opencv_param2: float | None = Field(None, ge=1.0, le=1000.0)
    opencv_min_radius: float | None = Field(None, ge=1.0, le=10000.0)
    opencv_max_radius: float | None = Field(None, ge=1.0, le=10000.0)

    @model_validator(mode="after")
    def validate_diameter_range(self):
        if self.min_diam > self.max_diam:
            raise ValueError("min_diam must be less than or equal to max_diam")
        if self.opencv_min_radius is not None and self.opencv_max_radius is not None and self.opencv_min_radius > self.opencv_max_radius:
            raise ValueError("opencv_min_radius must be less than or equal to opencv_max_radius")
        return self


class ContourMetricsRequest(BaseModel):
    input_source: str = Field("snake", pattern="^(snake|output1|output2|raw)$")
    output_slot: str = Field("2", pattern="^(1|2)$")
    points: list[SnakePoint] | None = Field(None, min_length=3)


class CannyComparisonRequest(CannyParameters):
    resize_to_notebook: bool = Field(
        True,
        description="Normalize the comparison input to the notebook MAX_DIM=400 canvas",
    )
    ellipse: EllipseParameters = Field(default_factory=EllipseParameters)


class PipelineRequest(BaseModel):
    canny: CannyParameters = Field(default_factory=CannyParameters)
    hough: HoughParameters = Field(default_factory=HoughParameters)
    hough_input: str = Field("canny", pattern="^(original|canny)$")


class ImageResponse(BaseModel):
    image_id: str
    width: int
    height: int
    channels: int


class ProcessResponse(BaseModel):
    image_id: str
    output: str
    opencv_output: str | None = None
    opencv_steps: dict[str, Any] | None = None
    steps: dict[str, Any]
    parameters: dict[str, Any]
