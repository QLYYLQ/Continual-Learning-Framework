from typing import Protocol
import httpx


class ImageAPIHandler(Protocol):
    base_url: str
    api_key: str

    def _build_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def validate(self) -> bool:
        """必须实现的验证方法"""
        ...

    async def fetch_image_urls(self, query: str) -> list[str]:
        """异步获取图片链接"""
        ...


class UnsplashHandler(ImageAPIHandler):
    base_url = "https://api.unsplash.com"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient()

    def validate(self):
        return APIKeyValidator.check_format(self.api_key)

    async def fetch_image_urls(self, query: str):
        endpoint = f"{self.base_url}/search/photos"
        resp = await self.client.get(
            endpoint,
            params={"query": query},
            headers=self._build_headers()
        )
        return [item["urls"]["regular"] for item in resp.json()["results"]]
