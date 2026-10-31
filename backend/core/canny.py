import numpy as np

ALLOWED_KERNEL_SIZES = (3, 5, 7, 9, 15)
SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
SOBEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def validate_parameters(kernel_size: int, sigma: float, high: float, low: float) -> None:
    if kernel_size not in ALLOWED_KERNEL_SIZES:
        raise ValueError(f"kernel_size must be one of {ALLOWED_KERNEL_SIZES}")
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")
    if high < low:
        raise ValueError("high_threshold must be greater than or equal to low_threshold")


def gaussian_kernel(image_dim: tuple[int, int], kernel_size: int, sigma: float) -> np.ndarray:
    height, width = image_dim
    if height < kernel_size or width < kernel_size:
        raise ValueError("The image is smaller than the selected kernel")
    if kernel_size not in ALLOWED_KERNEL_SIZES:
        raise ValueError(f"kernel_size must be one of {ALLOWED_KERNEL_SIZES}")
    center = kernel_size // 2
    x, y = np.meshgrid(
        np.linspace(-center, center, kernel_size),
        np.linspace(-center, center, kernel_size),
    )
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2 + 1e-5))
    return kernel / np.sum(kernel)


def convolution(image: list[list[float]], kernel: list[list[float]]) -> list[list[float]]:
    height, width = len(image), len(image[0])
    kernel_height, kernel_width = len(kernel), len(kernel[0])
    output = [[0.0] * (width - kernel_width + 1) for _ in range(height - kernel_height + 1)]
    for i in range(len(output)):
        for j in range(len(output[0])):
            total = 0.0
            for u in range(kernel_height):
                for v in range(kernel_width):
                    total += image[i + u][j + v] * kernel[kernel_height - 1 - u][kernel_width - 1 - v]
            output[i][j] = total
    return output


def _pad(image: np.ndarray, kernel_size: int) -> np.ndarray:
    return np.pad(image.astype(np.float64), kernel_size // 2, mode="reflect")


def gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel(image.shape, kernel_size, sigma)
    return np.array(convolution(_pad(image, kernel_size).tolist(), kernel.tolist()))


def apply_sobel(blurred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = _pad(blurred, 3)
    sobel_x = convolution(padded.tolist(), SOBEL_X.tolist())
    sobel_y = convolution(padded.tolist(), SOBEL_Y.tolist())
    x, y = np.array(sobel_x), np.array(sobel_y)
    return np.sqrt(x**2 + y**2), np.degrees(np.arctan2(y, x)) % 360


def non_maximum_suppression(gradient: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    height, width = gradient.shape
    suppressed = np.zeros_like(gradient)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = orientation[i, j] % 180
            if angle < 22.5 or angle >= 157.5:
                first, second = gradient[i, j - 1], gradient[i, j + 1]
            elif angle < 67.5:
                first, second = gradient[i - 1, j - 1], gradient[i + 1, j + 1]
            elif angle < 112.5:
                first, second = gradient[i - 1, j], gradient[i + 1, j]
            else:
                first, second = gradient[i - 1, j + 1], gradient[i + 1, j - 1]
            if gradient[i, j] >= max(first, second):
                suppressed[i, j] = gradient[i, j]
    return suppressed


def hysteresis_thresholding(image: np.ndarray, high: float, low: float) -> np.ndarray:
    result = np.zeros_like(image)
    stack: list[tuple[int, int]] = []
    strong = image >= high
    result[strong] = 255
    stack.extend(zip(*np.nonzero(strong)))
    while stack:
        row, column = stack.pop()
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                r, c = row + row_offset, column + column_offset
                if 0 <= r < image.shape[0] and 0 <= c < image.shape[1]:
                    if result[r, c] == 0 and image[r, c] >= low:
                        result[r, c] = 255
                        stack.append((r, c))
    return result


def canny_edge_detector(
    image: np.ndarray, kernel_size: int, sigma: float, high_threshold: float, low_threshold: float
) -> dict[str, np.ndarray]:
    validate_parameters(kernel_size, sigma, high_threshold, low_threshold)
    if image.ndim == 2:
        grayscale = image
    else:
        luminance = np.dot(image[..., :3].astype(np.float64), [0.299, 0.587, 0.114])
        grayscale = np.floor(luminance + 0.5).clip(0, 255).astype(np.uint8)
    blurred = gaussian_blur(grayscale, kernel_size, sigma)
    gradient, angles = apply_sobel(blurred)
    suppressed = non_maximum_suppression(gradient, angles)
    edges = hysteresis_thresholding(suppressed, high_threshold, low_threshold)
    return {
        "grayscale": grayscale,
        "gaussian_blur": blurred,
        "gradient_magnitude": gradient,
        "gradient_direction": angles,
        "non_maximum_suppression": suppressed,
        "edges": edges,
    }
