import logging
import os
from contextlib import contextmanager
from pathlib import Path

import vcr
from vcr.record_mode import RecordMode

logger = logging.getLogger(__name__)

CASSETTES_PATH = Path(__file__).resolve().parent.parent / "fixtures/cassettes"
VCR_DEFAULT_KWARGS = {
    "allow_playback_repeats": False,
    "decode_compressed_response": True,
    "filter_headers": [
        ("Authorization", "sanitized"),
        ("x-api-key", "sanitized"),
        ("set-cookie", None),
        ("cookie", None),
    ],
    "filter_post_data_parameters": [("api_key", "sanitized")],
}


def is_cassette_recording():
    return os.getenv("PROMPTLAYER_IS_CASSETTE_RECORDING", "false").lower() == "true"


@contextmanager
def assert_played(cassette_name, should_assert_played=True, play_count=None, **kwargs):
    combined_kwargs = VCR_DEFAULT_KWARGS | kwargs
    is_cassette_recording_ = is_cassette_recording()
    combined_kwargs.setdefault(
        "record_mode", RecordMode.ONCE.value if is_cassette_recording_ else RecordMode.NONE.value
    )

    vcr_instance = vcr.VCR()
    with vcr_instance.use_cassette(str(CASSETTES_PATH / cassette_name), **combined_kwargs) as cassette:
        yield cassette
        if should_assert_played and not is_cassette_recording_:
            if play_count is None:
                assert cassette.all_played, "Not all requests have played"
            else:
                actual_play_count = cassette.play_count
                if cassette.play_count != play_count:
                    play_counts = cassette.play_counts
                    for index, request in enumerate(cassette.requests):
                        logger.debug("%s played %s time(s)", request, play_counts[index])
                    raise AssertionError(f"Expected {play_count}, actually played {actual_play_count}")
