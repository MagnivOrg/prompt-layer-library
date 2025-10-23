from promptlayer.groups.groups import acreate, create


class GroupManager:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    def create(self):
        return create(self.api_key, self.base_url)


class AsyncGroupManager:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    async def create(self):
        return await acreate(self.api_key, self.base_url)


__all__ = ["GroupManager", "AsyncGroupManager"]
