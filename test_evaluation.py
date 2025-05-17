import pytest

from evaluation import make_paramset_string


@pytest.mark.parametrize("args,expected", [
    (dict(), ""),
    (dict(architecture="resnet50"), "architecture=resnet50"),
    (dict(augmentation=dict(enabled=True)), "augmentation.enabled=True"),
])
def test_make_paramset_string(args, expected):
    assert make_paramset_string(args) == expected
