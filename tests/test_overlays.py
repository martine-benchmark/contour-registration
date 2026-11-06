import numpy as np

from contour_reg.viz.overlays import draw_overlay, proportional_font_scale


def test_font_scale_grows_with_width():
    assert proportional_font_scale(4000) > proportional_font_scale(1000)


def test_overlay_returns_same_shape():
    img = np.zeros((480, 640, 3), np.uint8)
    out = draw_overlay(img, points=np.array([[10., 10.], [20., 20.]]), label="cl5_ch20")
    assert out.shape == img.shape


def test_contour_silhouette_overlay_colors_and_shape():
    import numpy as np

    from contour_reg.viz.overlays import contour_silhouette_overlay
    bg = np.full((100, 120), 200, np.uint8)
    contour = np.array([[10.0, 10.0]])
    silhouette = np.array([[80.0, 60.0]])
    out = contour_silhouette_overlay(bg, contour, silhouette, radius=1)
    assert out.shape == (100, 120, 3)
    assert tuple(int(x) for x in out[10, 10]) == (40, 40, 255)   # red contour (BGR)
    assert tuple(int(x) for x in out[60, 80]) == (0, 255, 0)     # green silhouette
    assert bg[10, 10] == 200                                     # input untouched


def test_contour_silhouette_overlay_crop_shrinks():
    import numpy as np

    from contour_reg.viz.overlays import contour_silhouette_overlay
    bg = np.zeros((1000, 1000), np.uint8)
    pts = np.array([[500.0, 500.0], [520.0, 520.0]])
    out = contour_silhouette_overlay(bg, pts, pts, crop=True)
    assert out.shape[0] < 1000 and out.shape[1] < 1000


def test_fp_fn_overlay_classifies_tp_fp():
    import numpy as np

    from contour_reg.viz.overlays import fp_fn_overlay
    bg = np.zeros((200, 200), np.uint8)
    silhouette = np.array([[50.0, 50.0], [60.0, 50.0]])
    contour = np.array([[50.0, 51.0],    # ~1px from S -> TP (green)
                        [150.0, 150.0]])  # far -> FP (red)
    out = fp_fn_overlay(bg, contour, silhouette, epsilon=2.0, radius=0, show_fn=False)
    assert out.shape == (200, 200, 3)
    assert tuple(int(x) for x in out[51, 50]) == (0, 255, 0)    # TP green
    assert tuple(int(x) for x in out[150, 150]) == (0, 0, 255)  # FP red


def test_fp_fn_overlay_crop_and_empty_contour():
    import numpy as np

    from contour_reg.viz.overlays import fp_fn_overlay
    bg = np.zeros((500, 500), np.uint8)
    out = fp_fn_overlay(bg, np.empty((0, 2)), np.array([[10.0, 10.0]]),
                        epsilon=2.0)  # no contour -> no crash
    assert out.shape == (500, 500, 3)
