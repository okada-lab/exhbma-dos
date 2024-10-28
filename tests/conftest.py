import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", default=False, help="run full test cases"
    )
    parser.addoption("--force-update", action="store_true", help="reset cache for test")


def pytest_configure(config):
    config.addinivalue_line("markers", "full: mark test as full test cases")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--full"):
        skip = pytest.mark.skip(reason="need --full option to run")
        for item in items:
            if "full" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def force_update(request):
    return request.config.getoption("--force-update")
