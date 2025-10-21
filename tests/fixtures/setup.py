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
