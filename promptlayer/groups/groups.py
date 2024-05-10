from promptlayer.utils import promptlayer_create_group


def create(api_key: str = None):
    """Create a new group."""
    return promptlayer_create_group(api_key)
