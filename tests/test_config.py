from contour_reg.config import CannyConfig, SweepGrid


def test_slug():
    assert CannyConfig(50, 150, 20, 3).slug == "cl50_ch150_m20_sk3"


def test_grid_filters_low_lt_high():
    cfgs = list(SweepGrid(lows=[5, 50], highs=[20, 10]).configs())
    assert all(c.canny_low < c.canny_high for c in cfgs)
    assert ("cl5_ch20" in {c.slug[:8] for c in cfgs})  # 5<20 kept; 50<10 dropped
