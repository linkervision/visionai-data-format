import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from visionai_data_format.converters import kitti_to_vai  # noqa: F401, E402
from visionai_data_format.utils.calculation import cart2hom  # noqa: E402
from visionai_data_format.utils.resize import resize_image  # noqa: E402


def test_cart2hom_appends_homogeneous_column():
    pcs_3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result = cart2hom(pcs_3d)

    assert result.tolist() == [[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]]


def test_resize_image_writes_resized_file(tmp_path):
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()

    cv2.imwrite(str(src_dir / "image.png"), np.zeros((20, 20, 3), dtype=np.uint8))

    resize_image(str(src_dir), str(dst_dir), (10, 10))

    resized = cv2.imread(str(dst_dir / "image.png"))
    assert resized.shape == (10, 10, 3)
