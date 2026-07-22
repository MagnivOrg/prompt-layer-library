from unittest.mock import patch

from promptlayer.evaluations.trace_price import wait_for_trace_request_price


def test_wait_for_trace_request_price_imports_when_price_appears():
    sleeps = []
    get_trace_responses = [
        {"success": True, "spans": [{"request_log_id": None}]},
        {"success": True, "spans": [{"request_log_id": 42}]},
        {"success": True, "spans": [{"request_log_id": 42}]},
    ]
    get_request_responses = [
        {"success": True, "price": None},
        {"success": True, "price": 0.001},
    ]

    with (
        patch(
            "promptlayer.evaluations.trace_price.tables_api.get_trace",
            side_effect=get_trace_responses,
        ) as mock_get_trace,
        patch(
            "promptlayer.evaluations.trace_price.tables_api.get_request",
            side_effect=get_request_responses,
        ) as mock_get_request,
    ):
        wait_for_trace_request_price(
            "key",
            "https://api.example.com",
            "abc123",
            delays_seconds=(0.1, 0.2, 0.2),
            sleep=sleeps.append,
        )

    assert sleeps == [0.1, 0.2, 0.2]
    assert mock_get_trace.call_count == 3
    assert mock_get_request.call_count == 2


def test_wait_for_trace_request_price_gives_up_after_max_wait():
    sleeps = []

    with (
        patch(
            "promptlayer.evaluations.trace_price.tables_api.get_trace",
            return_value={"success": True, "spans": []},
        ) as mock_get_trace,
        patch("promptlayer.evaluations.trace_price.tables_api.get_request") as mock_get_request,
    ):
        wait_for_trace_request_price(
            "key",
            "https://api.example.com",
            "abc123",
            max_wait_seconds=0.5,
            delays_seconds=(0.2, 0.2, 0.2),
            sleep=sleeps.append,
        )

    assert len(sleeps) == 3
    assert mock_get_trace.call_count == 3
    mock_get_request.assert_not_called()
