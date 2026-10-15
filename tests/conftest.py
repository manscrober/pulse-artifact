import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import golden  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture
def expected(request):
    """Reference digests for the test module, or a skip on platforms without fixtures."""
    module = request.module.__name__.removeprefix("test_")
    digests = golden.load(module)
    if digests is None:
        pytest.skip(f"no reference digests for {golden.platform_key()}")

    def check(name: str, tensor: torch.Tensor) -> None:
        assert golden.digest(tensor) == digests[name], name

    return check
