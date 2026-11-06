import numpy as np


class Camera:
    """Camera representation with intrinsic and extrinsic parameters."""

    def __init__(self, position: np.ndarray, rotation_world2cam: np.ndarray,
                 focal_length: float, principal_point: np.ndarray,
                 width: int, height: int,
                 distortion_coeffs: np.ndarray = None):
        """
        Initialize a camera from its parameters.

        Args:
            position: Camera center (3,)
            rotation_world2cam: world2cam rotation matrix (3x3)
            focal_length: Focal length in pixels
            principal_point: Principal point [cx, cy]
            width: Image width
            height: Image height
            distortion_coeffs: Radial distortion coefficients [k1, k2, k3] (optional)
        """
        self.position = np.array(position, dtype=np.float64)
        self.rotation_world2cam = np.array(rotation_world2cam, dtype=np.float64)
        self.focal_length = focal_length
        self.cx, self.cy = principal_point
        self.width = width
        self.height = height
        if distortion_coeffs is not None and len(distortion_coeffs) > 0:
            self.k1 = float(distortion_coeffs[0])
            self.k2 = float(distortion_coeffs[1]) if len(distortion_coeffs) > 1 else 0.0
            self.k3 = float(distortion_coeffs[2]) if len(distortion_coeffs) > 2 else 0.0
        else:
            self.k1 = self.k2 = self.k3 = 0.0

    @classmethod
    def from_look_at(cls, position: np.ndarray, look_at: np.ndarray, up: np.ndarray,
                     width: int, height: int, focal_length: float) -> 'Camera':
        """
        Create a camera from a position and a look-at point.

        Args:
            position: Camera position in the world frame [x, y, z]
            look_at: Point the camera looks at [x, y, z]
            up: Camera "up" vector [x, y, z]
            width: Image width in pixels
            height: Image height in pixels
            focal_length: Focal length in pixels

        Returns:
            Camera instance
        """
        position = np.array(position, dtype=np.float64)
        look_at = np.array(look_at, dtype=np.float64)
        up = np.array(up, dtype=np.float64)

        # Compute the camera basis vectors
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        up_corrected = np.cross(forward, right)

        # camera-to-world rotation matrix (columns = camera axes)
        R_c2w = np.column_stack([right, up_corrected, forward])

        # world-to-camera rotation matrix
        rotation_world2cam = R_c2w.T

        # Principal point at the image center
        principal_point = np.array([width / 2.0, height / 2.0])

        return cls(
            position=position,
            rotation_world2cam=rotation_world2cam,
            focal_length=focal_length,
            principal_point=principal_point,
            width=width,
            height=height
        )

    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points onto the image plane with optional radial distortion.

        Args:
            points_3d: Points 3D (N x 3)

        Returns:
            Projected 2D points (N x 2) with [u, v] coordinates
        """
        # World -> camera frame: P_cam = R * (P_world - C)
        points_cam = (self.rotation_world2cam @ (points_3d - self.position).T).T

        # Perspective projection
        Z = points_cam[:, 2]
        Z = np.where(np.abs(Z) < 1e-6, 1e-6, Z)

        # Normalized coordinates
        x = points_cam[:, 0] / Z
        y = points_cam[:, 1] / Z

        # Radial distortion (k1, k2, k3)
        if self.k1 != 0.0 or self.k2 != 0.0 or self.k3 != 0.0:
            r2 = x * x + y * y
            radial = 1.0 + self.k1 * r2 + self.k2 * (r2 * r2) + self.k3 * (r2 * r2 * r2)
            x = x * radial
            y = y * radial

        u = self.focal_length * x + self.cx
        v = self.focal_length * y + self.cy

        return np.column_stack([u, v])

    @property
    def forward(self) -> np.ndarray:
        """Return the camera forward vector."""
        return self.rotation_world2cam[2, :]

    @property
    def right(self) -> np.ndarray:
        """Return the camera right vector."""
        return self.rotation_world2cam[0, :]

    @property
    def up_corrected(self) -> np.ndarray:
        """Return the camera corrected up vector."""
        return self.rotation_world2cam[1, :]
