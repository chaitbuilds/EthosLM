#!/usr/bin/env python3
"""Auth-injecting HTTP proxy shim.

The sandbox has no DNS and only an authenticated upstream HTTP proxy. The JVM
(Fabric launcher, Chunky) reads -Dhttp.proxyHost but has no clean way to supply
proxy credentials. This listens without auth on 127.0.0.1 and re-emits every
request to the upstream proxy with a Proxy-Authorization header, so the JVM
never needs DNS or credentials.
"""
import base64
import os
import re
import select
import socket
import socketserver
import sys
import threading

UP = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""
m = re.match(r"https?://(?:([^:@]+):([^@]*)@)?([^:/]+):(\d+)", UP)
if not m:
    sys.exit(f"cannot parse upstream proxy from {UP!r}")
USER, PASS, UHOST, UPORT = m.group(1), m.group(2), m.group(3), int(m.group(4))
AUTH = base64.b64encode(f"{USER}:{PASS}".encode()).decode() if USER else None


def pump(a, b):
    try:
        while True:
            r, _, _ = select.select([a, b], [], [], 60)
            if not r:
                return
            for s in r:
                data = s.recv(65536)
                if not data:
                    return
                (b if s is a else a).sendall(data)
    except OSError:
        pass


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        cli = self.request
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = cli.recv(4096)
            if not chunk:
                return
            head += chunk
        header_blob, _, rest = head.partition(b"\r\n\r\n")
        lines = header_blob.split(b"\r\n")
        # strip any client-supplied proxy auth, add ours
        lines = [ln for ln in lines if not ln.lower().startswith(b"proxy-authorization")]
        if AUTH:
            lines.insert(1, f"Proxy-Authorization: Basic {AUTH}".encode())
        out = b"\r\n".join(lines) + b"\r\n\r\n" + rest

        up = socket.create_connection((UHOST, UPORT), timeout=30)
        try:
            up.sendall(out)
            pump(cli, up)
        finally:
            up.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    with Server(("127.0.0.1", port), Handler) as srv:
        print(f"authproxy on 127.0.0.1:{port} -> {UHOST}:{UPORT}", flush=True)
        srv.serve_forever()
