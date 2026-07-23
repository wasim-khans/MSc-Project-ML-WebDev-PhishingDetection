import ipaddress
import re
from urllib.parse import parse_qsl, urlsplit


FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "dot_count",
    "hyphen_count",
    "digit_count",
    "special_char_count",
    "has_https",
    "has_ip_address",
    "has_at_symbol",
    "subdomain_count",
    "query_param_count",
    "suspicious_word_count",
    "tld_length",
    "has_url_shortener",
]

SUSPICIOUS_WORDS = {
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "banking",
    "bank",
    "password",
    "confirm",
    "signin",
    "payment",
    "wallet",
    "auth",
    "recover",
    "support",
}

URL_SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
    "shorturl.at",
}

SPECIAL_CHARS = set("@%?=&_~#/:;,+!*'()[]")


def extract_features(url: str) -> dict[str, int]:
    cleaned_url = str(url).strip()
    parsed = _parse_url(cleaned_url)
    hostname = (parsed.hostname or "").lower()
    labels = [label for label in hostname.split(".") if label]
    tokens = [token for token in re.split(r"[^a-z0-9]+", cleaned_url.lower()) if token]
    has_ip_address = _is_ip_address(hostname)

    features = {
        "url_length": len(cleaned_url),
        "domain_length": len(hostname),
        "path_length": len(parsed.path or ""),
        "dot_count": cleaned_url.count("."),
        "hyphen_count": cleaned_url.count("-"),
        "digit_count": sum(char.isdigit() for char in cleaned_url),
        "special_char_count": sum(char in SPECIAL_CHARS for char in cleaned_url),
        "has_https": int(parsed.scheme.lower() == "https"),
        "has_ip_address": int(has_ip_address),
        "has_at_symbol": int("@" in cleaned_url),
        "subdomain_count": 0 if has_ip_address else max(len(labels) - 2, 0),
        "query_param_count": len(parse_qsl(parsed.query, keep_blank_values=True)),
        "suspicious_word_count": sum(token in SUSPICIOUS_WORDS for token in tokens),
        "tld_length": 0 if has_ip_address or not labels else len(labels[-1]),
        "has_url_shortener": int(hostname in URL_SHORTENER_DOMAINS),
    }

    return {name: features[name] for name in FEATURE_NAMES}


def _parse_url(url: str):
    parsed = urlsplit(url)
    if parsed.netloc:
        return parsed
    return urlsplit(f"//{url}")


def _is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True
