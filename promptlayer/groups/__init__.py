from promptlayer.groups.groups import acreate, create


class GroupManager:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def create(self):
        return create(self.api_key)


class AsyncGroupManager:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def create(self) -> str:
        return await acreate(self.api_key)


__all__ = ["GroupManager", "AsyncGroupManager"]
