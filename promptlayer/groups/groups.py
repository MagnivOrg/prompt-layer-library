from promptlayer.utils import apromptlayer_create_group, promptlayer_create_group


def create(api_key: str, base_url: str, throw_on_error: bool):
    return promptlayer_create_group(api_key, base_url, throw_on_error)


async def acreate(api_key: str, base_url: str, throw_on_error: bool):
    return await apromptlayer_create_group(api_key, base_url, throw_on_error)
