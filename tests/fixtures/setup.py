from unittest.mock import patch

import pytest

from tests.utils.vcr import is_cassette_recording

if is_cassette_recording():

    @pytest.fixture
    def autouse_disable_network():
        return
else:

    @pytest.fixture(autouse=True)
    def autouse_disable_network(disable_network):
        yield


@pytest.fixture(scope="session", autouse=True)
def setup():
    with patch("promptlayer.utils.URL_API_PROMPTLAYER", "http://localhost:8000"):
        yield
