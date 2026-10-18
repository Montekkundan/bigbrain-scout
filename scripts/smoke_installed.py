"""Verify an installed Scout in a clean working directory, without extra packages.

Usage: python scripts/smoke_installed.py --python /venv/bin/python
The interpreter must already contain the built Scout wheel, not an editable install.
"""

import argparse
import http.client
import io
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import tempfile
import threading
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    # Do not resolve the Python symlink: it carries the venv identity.
    python = str(Path(args.python).absolute())
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="scout-installed-") as temporary:
        root = Path(temporary).resolve()

        def run(*arguments, expected=0):
            result = subprocess.run([python, "-I", "-m", "bigbrain_scout", *map(str, arguments)],
                                    cwd=root, env=environment, text=True, capture_output=True, timeout=30)
            if result.returncode != expected:
                raise RuntimeError(f"Installed CLI {arguments} returned {result.returncode}: {result.stderr}")
            return json.loads(result.stdout) if result.stdout else None

        assert run("plan", "--example", "safe")["shape"] == [2, 3, 1]
        run("plan", "--example", "unsafe", expected=2)
        assert [item["id"] for item in run("sources")["sources"]] == ["section-3905", "section-3954", "section-4003"]
        model = run("model-plan")
        assert model["on_hold"] and not model["inference_executed"]
        assert model["checkpoint"]["training_section_ids"] is None
        run("infer", "--audit", root / "unused-audit", "--checkpoint", root / "unused-checkpoint",
            "--output", root / "refused-model", expected=2)
        assert not (root / "refused-model").exists()
        run("demo", "--output", root / "original")
        assert run("verify", "--capsule", root / "original")["verified"]
        assert run("reproduce", "--capsule", root / "original", "--output", root / "replayed")["comparison"]["matched"]
        assert run("compare", "--first", root / "original", "--second", root / "replayed")["data_bytes_equal"]
        run("demo", "--output", root / "original", expected=2)
        run("fetch", "--source", "section-3954", "--output", root / "refused-live", expected=2)
        assert not (root / "refused-live").exists()
        entry = Path(args.python).absolute().parent / ("bigbrain-scout.exe" if os.name == "nt" else "bigbrain-scout")
        result = subprocess.run([str(entry), "--help"], cwd=root, env=environment,
                                text=True, capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert "reproduce" in result.stdout
        checked = run("check", "--output", root / "teammate")
        assert checked["passed"] and checked["synthetic"]["matched"]
        assert not checked["real_audit"]["executed"] and not checked["gpu_used"]
        assert json.loads((root / "teammate" / "check.json").read_bytes()) == checked
        run("check", "--output", root / "teammate", expected=2)

        server = subprocess.Popen([python, "-I", "-m", "bigbrain_scout", "serve", "--port", "0", "--output", str(root / "workspace")],
                                  cwd=root, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        lines = queue.Queue()
        reader = threading.Thread(target=lambda: lines.put(server.stdout.readline()), daemon=True)
        reader.start()
        try:
            first = lines.get(timeout=15)
            match = re.fullmatch(r"BigBrain Scout: http://127\.0\.0\.1:(\d+)\n", first)
            if not match:
                raise RuntimeError(f"Installed server failed to start: {first}")
            port = int(match.group(1))

            def call(method, route, body=None, token=None):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                try:
                    headers = {"Content-Type": "application/json"}
                    if token:
                        headers["X-Scout-Token"] = token
                    payload = json.dumps(body).encode() if body is not None else None
                    connection.request(method, route, payload, headers)
                    response = connection.getresponse()
                    data = response.read()
                    assert response.status == 200, (response.status, data)
                    return data
                finally:
                    connection.close()

            assert b"Small regions." in call("GET", "/")
            assert b"function sourceChanged" in call("GET", "/app.js")
            assert b"--teal" in call("GET", "/style.css")
            config = json.loads(call("GET", "/api/config"))
            assert not config["allow_live"]
            token = config["csrf_token"]
            checked = json.loads(call("POST", "/api/plan", config["sources"][0]["defaults"], token))
            capsule = json.loads(call("POST", "/api/build", {"token": checked["token"]}, token))
            assert json.loads(call("POST", "/api/verify", {"id": capsule["id"]}, token))["verified"]
            payload = call("GET", f'/api/capsules/{capsule["id"]}/download')
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                assert set(archive.namelist()) == {"manifest.json", "data.raw", "preview.pgm", "fixture.json", "reproduce.py"}
                archive.extractall(root / "downloaded")
            assert run("verify", "--capsule", root / "downloaded")["verified"]
        finally:
            server.terminate()
            server.communicate(timeout=10)
            reader.join(timeout=1)
    print("Installed wheel: console CLI, defaults, replay, comparison, held model metadata/opt-in, HTTP assets, capsule export and refusal checks passed")


if __name__ == "__main__":
    main()
