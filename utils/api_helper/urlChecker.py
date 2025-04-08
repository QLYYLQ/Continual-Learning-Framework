from urllib.parse import urlparse
import ipaddress


def validate_url(url: str):
    parsed = urlparse(url)
    # 禁止非HTTP协议
    if parsed.scheme not in ('http', 'https'):
        raise ValueError("Invalid protocol")

    # 解析IP地址
    try:
        ip = ipaddress.ip_address(parsed.hostname)
        # 禁止内网IP
        if ip.is_private:
            raise ValueError("Internal network access forbidden")
    except:
        pass  # 域名无需处理
    finally:
        pass

