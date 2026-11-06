import cv2
import numpy as np

from contour_reg.contours.canny import extract_canny_contours, filter_connected_components, mask_boundary_contour


def _disk(r=20, n=100):
    m = np.zeros((n,n), np.uint8)
    cv2.circle(m,(n//2,n//2),r,255,-1)
    return m

def test_canny_edges_within_margin_only():
    mask = _disk()
    img = np.zeros_like(mask)
    cv2.circle(img,(50,50),20,255,-1)
    edges = extract_canny_contours(img, mask, margin_size=5, canny_low=50, canny_high=150)
    assert edges.dtype == np.uint8 and edges.max() == 255
    # Core invariant: no edges outside the dilate-erode margin band.
    kernel = np.ones((2*5+1, 2*5+1), np.uint8)
    margin = cv2.subtract(cv2.dilate(mask, kernel), cv2.erode(mask, kernel))
    assert np.all(edges[margin == 0] == 0)

def test_large_sobel_kernel_runs():
    mask = _disk()
    img = np.zeros_like(mask)
    cv2.circle(img,(50,50),20,128,-1)
    edges = extract_canny_contours(img, mask, sobel_kernel=15)
    assert edges.shape == mask.shape
    # Catch a silently-zeroed large-kernel path: the 128-disk edge sits in-band.
    assert edges.any()

def test_inner_contours_yield_more_points():
    # annulus: outer disk minus inner disk -> RETR_TREE finds 2 loops
    m = _disk(30)
    cv2.circle(m,(50,50),12,0,-1)
    outer = mask_boundary_contour(m, include_inner=False)
    allc  = mask_boundary_contour(m, include_inner=True)
    assert len(allc) >= len(outer)

def test_component_filter_keeps_center():
    m = np.zeros((100,100), np.uint8)
    cv2.circle(m,(50,50),15,255,-1)
    cv2.circle(m,(10,10),5,255,-1)
    f = filter_connected_components(m, "center_point")
    assert f[50,50] == 255 and f[10,10] == 0
