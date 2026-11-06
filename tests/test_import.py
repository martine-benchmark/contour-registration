def test_package_imports():
    import contour_reg
    assert hasattr(contour_reg, "__version__")
