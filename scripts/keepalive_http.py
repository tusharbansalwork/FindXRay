"""Tiny HTTP client that reuses one connection, like a browser does (stdlib only).

urllib opens a new TCP + TLS connection for every request. Against a far-away region that adds several
round trips per request (measured: 1.8 s vs 0.43 s from Jaipur to Sydney), which is not what a visitor's
browser experiences. Scripts that measure latency use this instead.
"""

from __future__ import annotations

import http.client
import json
from urllib.parse import urlsplit


class KeepAlive:
    def __init__(self, base_url: str, timeout: float = 120):
        u = urlsplit(base_url.rstrip("/"))
        self.https, self.host, self.port = u.scheme == "https", u.hostname, u.port
        self.prefix, self.timeout, self.conn = u.path, timeout, None

    def _connect(self):
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        self.conn = cls(self.host, self.port, timeout=self.timeout)

    def request(self, method: str, path: str, body: dict | None = None, headers: dict | None = None):
        """Returns (status, headers as a lower-case dict, body bytes). Reconnects once if the server closed the connection."""
        data = json.dumps(body).encode() if body is not None else None
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        for attempt in (1, 2):
            if self.conn is None:
                self._connect()
            try:
                self.conn.request(method, self.prefix + path, body=data, headers=hdrs)
                r = self.conn.getresponse()
                raw = r.read()
                return r.status, {k.lower(): v for k, v in r.getheaders()}, raw
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError, http.client.CannotSendRequest):
                self.conn.close()
                self.conn = None
                if attempt == 2:
                    raise

    def post_json(self, path: str, body: dict) -> dict:
        status, _, raw = self.request("POST", path, body)
        if status != 200:
            raise RuntimeError(f"POST {path} -> HTTP {status}: {raw[:200]!r}")
        return json.loads(raw)

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
