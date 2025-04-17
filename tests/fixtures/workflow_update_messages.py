import pytest


@pytest.fixture
def workflow_update_data_no_result_code():
    return {
        "final_output": {
            "Node 1": {
                "status": "SUCCESS",
                "value": "no_result_code",
                "error_message": None,
                "raw_error_message": None,
                "is_output_node": None,
            }
        },
        "workflow_version_execution_id": 717,
    }


@pytest.fixture
def workflow_update_data_ok():
    return {
        "final_output": {
            "Node 1": {
                "status": "SUCCESS",
                "value": "ok_result_code",
                "error_message": None,
                "raw_error_message": None,
                "is_output_node": None,
            }
        },
        "result_code": "OK",
        "workflow_version_execution_id": 717,
    }


@pytest.fixture
def workflow_update_data_exceeds_size_limit():
    return {
        "final_output": (
            "Final output (and associated metadata) exceeds the size limit of 1 bytes. "
            "Upgrade to the most recent SDK or use GET /workflow-version-execution-results "
            "to retrieve the final output."
        ),
        "result_code": "EXCEEDS_SIZE_LIMIT",
        "workflow_version_execution_id": 717,
    }
