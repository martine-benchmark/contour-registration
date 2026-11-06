from dataclasses import dataclass
from typing import Iterator, List


@dataclass
class CannyConfig:
    canny_low: int
    canny_high: int
    margin_size: int = 20
    sobel_kernel: int = 3
    component_mode: str = "hybrid"
    include_inner_contours: bool = False

    @property
    def slug(self) -> str:
        return f"cl{self.canny_low}_ch{self.canny_high}_m{self.margin_size}_sk{self.sobel_kernel}"


@dataclass
class ICPConfig:
    max_iterations: int = 50
    error_percentile: float = 75.0
    max_correspondence_distance: float = 0.0
    keep_pct: float = 100.0
    min_canny_points: int = 0
    include_inner_contours: bool = False
    dof: int = 6


@dataclass
class SweepGrid:
    """Canny thresholds to sweep (pairs with ``low < high``); Sobel kernel and margin are shared."""

    lows: List[int]
    highs: List[int]
    sobel_kernel: int = 3
    margin_size: int = 20

    def configs(self) -> Iterator[CannyConfig]:
        for low in self.lows:
            for high in self.highs:
                if low < high:
                    yield CannyConfig(canny_low=low, canny_high=high, margin_size=self.margin_size,
                                      sobel_kernel=self.sobel_kernel)
