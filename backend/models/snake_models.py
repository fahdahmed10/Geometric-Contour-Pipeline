from pydantic import BaseModel, Field


class SnakePoint(BaseModel):
    x: float
    y: float


class SnakeRequest(BaseModel):
    initial_contour: list[SnakePoint] = Field(..., min_length=3)
    output_slot: str = Field("2", pattern="^(1|2)$")
    alpha: float = Field(1.0, ge=0)
    beta: float = Field(1.0, ge=0)
    gamma: float = Field(1.2, ge=0)
    window_size: int = Field(5, ge=3, le=11)
    max_iterations: int = Field(200, ge=1, le=1000)
    movement_threshold: int = Field(1, ge=0)
    gaussian_sigma: float = Field(1.0, ge=0)
    corner_relaxation: bool = True
    corner_threshold: float = Field(0.1, ge=0)
    resample_every_k: int = Field(0, ge=0)
    n_points: int = Field(20, ge=3, le=200)
