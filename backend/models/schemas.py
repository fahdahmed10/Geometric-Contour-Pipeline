from typing import Any

from pydantic import BaseModel, Field


class CannyParameters(BaseModel):
    kernel_size: int = Field(7, description="Odd kernel size supported by the notebook")
    sigma: float = Field(1.0, gt=0)
    low_threshold: float = Field(80.0, ge=0)
    high_threshold: float = Field(200.0, ge=0)


class HoughParameters(BaseModel):
    radius_res: float = Field(1.0, gt=0)
    theta_res: float = Field(1.0, gt=0, le=180)
    max_lines: int = Field(10, ge=1, le=100)
    vote_threshold: int = Field(20, ge=0)


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
    steps: dict[str, Any]
    parameters: dict[str, Any]
