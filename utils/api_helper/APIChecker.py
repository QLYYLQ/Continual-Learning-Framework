import requests
import re
from requests.exceptions import ConnectionError
from itsdangerous import TimestampSigner


def test_api_endpoint(url: str, timeout=5):
    try:
        resp = requests.head(url, timeout=timeout)
        return resp.status_code == 200
    except ConnectionError:
        return False


class APIKeyValidator:
    # 基础格式校验
    @staticmethod
    def check_format(api_key: str):
        return re.match(r'^[a-f0-9]{32}$', api_key) is not None

    # 签名验证（需与服务器一致）
    def verify_signature(self, api_key: str, secret: str):
        s = TimestampSigner(secret)
        try:
            s.unsign(api_key, max_age=3600 * 24)
            return True
        except:
            return False
