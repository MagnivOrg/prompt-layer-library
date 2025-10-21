from promptlayer.utils import apromptlayer_create_group, promptlayer_create_group


def create(api_key: str, base_url: str):
    return promptlayer_create_group(api_key, base_url)


async def acreate(api_key: str, base_url: str):
    return await apromptlayer_create_group(api_key, base_url)
