from promptlayer.groups.groups import acreate, create


class GroupManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error

    def create(self):
        return create(self.api_key, self.base_url, self.throw_on_error)


class AsyncGroupManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error

    async def create(self):
        return await acreate(self.api_key, self.base_url, self.throw_on_error)


__all__ = ["GroupManager", "AsyncGroupManager"]
