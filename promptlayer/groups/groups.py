from promptlayer.utils import apromptlayer_create_group, promptlayer_create_group


def create(api_key: str = None):
    """Create a new group."""
    return promptlayer_create_group(api_key)


async def acreate(api_key: str = None) -> str:
    """Asynchronously create a new group."""
    return await apromptlayer_create_group(api_key)
