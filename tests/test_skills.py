import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptlayer import PromptLayerValidationError
from promptlayer.skills import (
    acreate_skill_collection,
    apull_skill_collection,
    asave_skill_collection_version,
    aupdate_skill_collection,
    create_skill_collection,
    pull_skill_collection,
    save_skill_collection_version,
)


def test_sync_pull_skill_collection_encodes_identifier_and_forwards_params(promptlayer_api_key, base_url):
    identifier = "folder/skill set"
    expected_encoded = "folder%2Fskill%20set"
    payload = {
        "success": True,
        "skill_collection": {"id": "collection-id", "name": "folder/skill set"},
        "files": {"docs/SKILL.md": "hello"},
        "version": {"number": 2},
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = payload

    with patch("promptlayer.skills.api._get_requests_session") as mock_session:
        mock_session.return_value.get.return_value = mock_response

        response = pull_skill_collection(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            identifier=identifier,
            label="prod",
            version=2,
        )

        call_args = mock_session.return_value.get.call_args
        actual_url = call_args[0][0]
        assert expected_encoded in actual_url, f"Expected {expected_encoded} in URL, got {actual_url}"
        assert call_args[1]["params"] == {"label": "prod", "version": 2}
        assert response == payload


@pytest.mark.asyncio
async def test_async_pull_skill_collection_supports_zip_format(promptlayer_api_key, base_url):
    identifier = "folder/skill set"
    expected_encoded = "folder%2Fskill%20set"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"zip-bytes"

    with patch("promptlayer.skills.api._make_httpx_client") as mock_client_factory:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_factory.return_value = mock_client

        response = await apull_skill_collection(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            identifier=identifier,
            format="zip",
        )

        call_args = mock_client.get.call_args
        actual_url = call_args[0][0]
        assert expected_encoded in actual_url, f"Expected {expected_encoded} in URL, got {actual_url}"
        assert call_args[1]["params"] == {"format": "zip"}
        assert response == b"zip-bytes"


def test_sync_create_skill_collection_posts_expected_body(promptlayer_api_key, base_url):
    body = {
        "name": "SDK Skills",
        "files": [{"path": "docs/SKILL.md", "content": "hello"}],
        "commit_message": "seed skills",
    }
    payload = {
        "success": True,
        "skill_collection": {"id": "collection-id", "name": "SDK Skills"},
    }

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = payload

    with patch("promptlayer.skills.api._get_requests_session") as mock_session:
        mock_session.return_value.post.return_value = mock_response

        response = create_skill_collection(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            body=body,
        )

        call_args = mock_session.return_value.post.call_args
        assert call_args[0][0] == f"{base_url}/api/public/v2/skill-collections"
        assert call_args[1]["json"] == body
        assert response == payload


def test_skill_manager_publish_requires_provider(promptlayer_client):
    with pytest.raises(PromptLayerValidationError, match="provider"):
        promptlayer_client.skills.publish({"name": "SDK Skills"})


def test_skill_manager_publish_rejects_invalid_provider(promptlayer_client):
    with pytest.raises(PromptLayerValidationError, match="claude_code, openai, openclaw"):
        promptlayer_client.skills.publish({"name": "SDK Skills", "provider": "anthropic"})


def test_skill_manager_publish_delegates_to_create(promptlayer_client):
    payload = {
        "success": True,
        "skill_collection": {"id": "collection-id", "name": "SDK Skills"},
    }
    body = {"name": "SDK Skills", "provider": "openai"}

    with patch("promptlayer.skills.manager.create_skill_collection", return_value=payload) as mock_create:
        response = promptlayer_client.skills.publish(body)

        mock_create.assert_called_once_with(
            promptlayer_client.api_key,
            promptlayer_client.base_url,
            promptlayer_client.throw_on_error,
            body,
        )
        assert response == payload


def test_sync_create_skill_collection_supports_zip_upload(promptlayer_api_key, base_url):
    body = {"name": "SDK Skills", "commit_message": "seed from zip"}
    zip_bytes = b"PK\x03\x04demo"
    payload = {
        "success": True,
        "skill_collection": {"id": "collection-id", "name": "SDK Skills"},
    }

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = payload

    with patch("promptlayer.skills.api._get_requests_session") as mock_session:
        mock_session.return_value.post.return_value = mock_response

        response = create_skill_collection(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            body=body,
            zip=zip_bytes,
        )

        call_args = mock_session.return_value.post.call_args
        assert call_args[0][0] == f"{base_url}/api/public/v2/skill-collections"
        assert call_args[1]["data"] == {"metadata": json.dumps(body)}
        assert call_args[1]["files"]["zip"] == ("skills.zip", zip_bytes, "application/zip")
        assert response == payload


@pytest.mark.asyncio
async def test_async_update_skill_collection_patches_encoded_identifier(promptlayer_api_key, base_url):
    identifier = "folder/skill set"
    expected_encoded = "folder%2Fskill%20set"
    payload = {
        "success": True,
        "skill_collection": {"id": "collection-id", "name": "Renamed Skills"},
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = payload

    with patch("promptlayer.skills.api._make_httpx_client") as mock_client_factory:
        mock_client = AsyncMock()
        mock_client.patch.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_factory.return_value = mock_client

        response = await aupdate_skill_collection(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            identifier=identifier,
            body={"name": "Renamed Skills"},
        )

        call_args = mock_client.patch.call_args
        actual_url = call_args[0][0]
        assert expected_encoded in actual_url, f"Expected {expected_encoded} in URL, got {actual_url}"
        assert call_args[1]["json"] == {"name": "Renamed Skills"}
        assert response == payload


def test_sync_save_skill_collection_version_supports_zip_upload(promptlayer_api_key, base_url):
    body = {"commit_message": "zip update"}
    zip_bytes = b"PK\x03\x04demo"
    payload = {"success": True, "version": {"number": 2}}

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = payload

    with patch("promptlayer.skills.api._get_requests_session") as mock_session:
        mock_session.return_value.post.return_value = mock_response

        response = save_skill_collection_version(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            identifier="collection-id",
            body=body,
            zip=zip_bytes,
        )

        call_args = mock_session.return_value.post.call_args
        assert call_args[0][0] == f"{base_url}/api/public/v2/skill-collections/collection-id/versions"
        assert call_args[1]["data"] == {"metadata": json.dumps(body)}
        assert call_args[1]["files"]["zip"] == ("skills.zip", zip_bytes, "application/zip")
        assert response == payload


def test_skill_manager_update_combines_rename_and_version_save(promptlayer_client):
    rename_response = {
        "success": True,
        "skill_collection": {"id": "collection-id", "name": "Renamed Skills"},
    }
    version_response = {
        "success": True,
        "version": {"number": 3, "commit_message": "save update"},
    }

    with (
        patch("promptlayer.skills.manager.update_skill_collection", return_value=rename_response) as mock_rename,
        patch(
            "promptlayer.skills.manager.save_skill_collection_version", return_value=version_response
        ) as mock_save_version,
        patch("promptlayer.skills.manager.pull_skill_collection") as mock_pull,
    ):
        response = promptlayer_client.skills.update(
            "Original Skills",
            name="Renamed Skills",
            file_updates=[{"path": "docs/SKILL.md", "content": "hello"}],
            commit_message="save update",
        )

        mock_rename.assert_called_once_with(
            promptlayer_client.api_key,
            promptlayer_client.base_url,
            promptlayer_client.throw_on_error,
            "Original Skills",
            {"name": "Renamed Skills"},
        )
        mock_save_version.assert_called_once_with(
            promptlayer_client.api_key,
            promptlayer_client.base_url,
            promptlayer_client.throw_on_error,
            "collection-id",
            {
                "file_updates": [{"path": "docs/SKILL.md", "content": "hello"}],
                "commit_message": "save update",
            },
        )
        mock_pull.assert_not_called()
        assert response == {
            "success": True,
            "skill_collection": {"id": "collection-id", "name": "Renamed Skills"},
            "version": {"number": 3, "commit_message": "save update"},
        }


@pytest.mark.asyncio
async def test_async_skill_manager_update_fetches_collection_for_version_only(promptlayer_async_client):
    version_response = {
        "success": True,
        "version": {"number": 4, "commit_message": "save update"},
    }
    pull_response = {
        "success": True,
        "skill_collection": {"id": "collection-id", "name": "Current Skills"},
        "files": {"docs/SKILL.md": "hello"},
        "version": {"number": 4},
    }

    with (
        patch("promptlayer.skills.manager.aupdate_skill_collection") as mock_rename,
        patch(
            "promptlayer.skills.manager.asave_skill_collection_version", return_value=version_response
        ) as mock_save_version,
        patch("promptlayer.skills.manager.apull_skill_collection", return_value=pull_response) as mock_pull,
    ):
        response = await promptlayer_async_client.skills.update(
            "collection-id",
            file_updates=[{"path": "docs/SKILL.md", "content": "hello"}],
            release_label="prod",
        )

        mock_rename.assert_not_called()
        mock_save_version.assert_called_once_with(
            promptlayer_async_client.api_key,
            promptlayer_async_client.base_url,
            promptlayer_async_client.throw_on_error,
            "collection-id",
            {
                "file_updates": [{"path": "docs/SKILL.md", "content": "hello"}],
                "release_label": "prod",
            },
        )
        mock_pull.assert_called_once_with(
            promptlayer_async_client.api_key,
            promptlayer_async_client.base_url,
            promptlayer_async_client.throw_on_error,
            "collection-id",
            label=None,
            version=None,
            format=None,
        )
        assert response == {
            "success": True,
            "skill_collection": {"id": "collection-id", "name": "Current Skills"},
            "version": {"number": 4, "commit_message": "save update"},
        }


def test_skill_manager_update_supports_zip_only(promptlayer_client):
    version_response = {
        "success": True,
        "version": {"number": 4, "commit_message": "zip update"},
    }
    pull_response = {
        "success": True,
        "skill_collection": {"id": "collection-id", "name": "Current Skills"},
        "files": {"docs/SKILL.md": "hello"},
        "version": {"number": 4},
    }
    zip_bytes = b"PK\x03\x04demo"

    with (
        patch("promptlayer.skills.manager.update_skill_collection") as mock_rename,
        patch(
            "promptlayer.skills.manager.save_skill_collection_version", return_value=version_response
        ) as mock_save_version,
        patch("promptlayer.skills.manager.pull_skill_collection", return_value=pull_response) as mock_pull,
    ):
        response = promptlayer_client.skills.update("collection-id", zip=zip_bytes)

        mock_rename.assert_not_called()
        mock_save_version.assert_called_once_with(
            promptlayer_client.api_key,
            promptlayer_client.base_url,
            promptlayer_client.throw_on_error,
            "collection-id",
            {},
            zip=zip_bytes,
        )
        mock_pull.assert_called_once()
        assert response == {
            "success": True,
            "skill_collection": {"id": "collection-id", "name": "Current Skills"},
            "version": {"number": 4, "commit_message": "zip update"},
        }


def test_skill_manager_update_requires_changes(promptlayer_client):
    with pytest.raises(PromptLayerValidationError):
        promptlayer_client.skills.update("collection-id")


@pytest.mark.asyncio
async def test_async_create_skill_collection_posts_expected_body(promptlayer_api_key, base_url):
    body = {"name": "Async SDK Skills"}
    payload = {
        "success": True,
        "skill_collection": {"id": "collection-id", "name": "Async SDK Skills"},
    }

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = payload

    with patch("promptlayer.skills.api._make_httpx_client") as mock_client_factory:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_factory.return_value = mock_client

        response = await acreate_skill_collection(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            body=body,
        )

        call_args = mock_client.post.call_args
        assert call_args[0][0] == f"{base_url}/api/public/v2/skill-collections"
        assert call_args[1]["json"] == body
        assert response == payload


@pytest.mark.asyncio
async def test_async_skill_manager_publish_requires_provider(promptlayer_async_client):
    with pytest.raises(PromptLayerValidationError, match="provider"):
        await promptlayer_async_client.skills.publish({"name": "Async SDK Skills"})


@pytest.mark.asyncio
async def test_async_skill_manager_publish_rejects_invalid_provider(promptlayer_async_client):
    with pytest.raises(PromptLayerValidationError, match="claude_code, openai, openclaw"):
        await promptlayer_async_client.skills.publish({"name": "Async SDK Skills", "provider": "anthropic"})


@pytest.mark.asyncio
async def test_async_skill_manager_publish_delegates_to_create(promptlayer_async_client):
    payload = {
        "success": True,
        "skill_collection": {"id": "collection-id", "name": "Async SDK Skills"},
    }
    body = {"name": "Async SDK Skills", "provider": "openai"}

    with patch("promptlayer.skills.manager.acreate_skill_collection", return_value=payload) as mock_create:
        response = await promptlayer_async_client.skills.publish(body)

        mock_create.assert_called_once_with(
            promptlayer_async_client.api_key,
            promptlayer_async_client.base_url,
            promptlayer_async_client.throw_on_error,
            body,
        )
        assert response == payload


@pytest.mark.asyncio
async def test_async_create_skill_collection_supports_zip_upload(promptlayer_api_key, base_url):
    body = {"name": "Async SDK Skills", "commit_message": "seed from zip"}
    zip_bytes = b"PK\x03\x04demo"
    payload = {
        "success": True,
        "skill_collection": {"id": "collection-id", "name": "Async SDK Skills"},
    }

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = payload

    with patch("promptlayer.skills.api._make_httpx_client") as mock_client_factory:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_factory.return_value = mock_client

        response = await acreate_skill_collection(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            body=body,
            zip=zip_bytes,
        )

        call_args = mock_client.post.call_args
        assert call_args[0][0] == f"{base_url}/api/public/v2/skill-collections"
        assert call_args[1]["data"] == {"metadata": json.dumps(body)}
        assert call_args[1]["files"]["zip"] == ("skills.zip", zip_bytes, "application/zip")
        assert response == payload


@pytest.mark.asyncio
async def test_async_save_skill_collection_version_supports_zip_upload(promptlayer_api_key, base_url):
    body = {"commit_message": "zip update"}
    zip_bytes = b"PK\x03\x04demo"
    payload = {"success": True, "version": {"number": 2}}

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = payload

    with patch("promptlayer.skills.api._make_httpx_client") as mock_client_factory:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_factory.return_value = mock_client

        response = await asave_skill_collection_version(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            identifier="collection-id",
            body=body,
            zip=zip_bytes,
        )

        call_args = mock_client.post.call_args
        assert call_args[0][0] == f"{base_url}/api/public/v2/skill-collections/collection-id/versions"
        assert call_args[1]["data"] == {"metadata": json.dumps(body)}
        assert call_args[1]["files"]["zip"] == ("skills.zip", zip_bytes, "application/zip")
        assert response == payload
