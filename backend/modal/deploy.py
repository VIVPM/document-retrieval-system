"""
Deploys the Docling worker and proves it actually serves.

`modal deploy` alone cannot be trusted here. Two failures both end with a
zero exit code and nothing working:

  - On a Windows cp1252 console the CLI's progress rendering dies with
    "'charmap' codec can't encode" AFTER the image builds, and exits 0.
  - An image can build and deploy clean and still fail every request at
    runtime (a missing shared library, a bad model path).

So this forces UTF-8, requires the deployed URL to appear in the output, and
then puts a real PDF through /extract before reporting success.

Usage:
    python backend/modal/deploy.py [--pipeline classic|vlm] [--pdf PATH]
"""
import argparse
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
REPO = os.path.dirname(BACKEND)
WORKER = os.path.join(HERE, "modal_docling_worker.py")
URL_RE = re.compile(r"https://[^\s]+\.modal\.run")


def modal_cmd() -> list:
    """Locate the modal CLI: PATH, then the repo venv, then this interpreter."""
    from shutil import which
    found = which("modal")
    if found:
        return [found]
    venv = os.path.join(BACKEND, ".venv", "Scripts", "modal.exe")
    if os.path.exists(venv):
        return [venv]
    return [sys.executable, "-m", "modal"]


def deploy() -> str:
    """Run `modal deploy` and return the deployed URL."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
           "TERM": "dumb"}
    proc = subprocess.run(
        modal_cmd() + ["deploy", WORKER],
        env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-2000:])

    if "charmap" in out:
        sys.exit("FAILED: console encoding killed the CLI. UTF-8 was not applied.")

    urls = URL_RE.findall(out)
    if not urls:
        sys.exit(f"FAILED: no .modal.run URL in the output (exit={proc.returncode}). "
                 "Nothing was deployed, whatever the exit code says.")
    return urls[0].rstrip("/")


def smoke(url: str, pipeline: str, pdf: str) -> None:
    """POST a real PDF and require extracted content back."""
    import requests

    print(f"[smoke] {pipeline} pipeline against {url}/extract")
    with open(pdf, "rb") as f:
        r = requests.post(
            f"{url}/extract",
            files={"file": (os.path.basename(pdf), f, "application/pdf")},
            params={"pipeline": pipeline},
            timeout=1800 if pipeline == "vlm" else 600,
        )
    if r.status_code != 200:
        sys.exit(f"FAILED: /extract returned {r.status_code}: {r.text[:400]}")

    data = r.json()
    if not data.get("success"):
        sys.exit(f"FAILED: extraction error: {data.get('error')}")

    chars = sum(len(b["content"])
                for items in data["pages"].values() for b in items)
    if chars == 0:
        sys.exit("FAILED: worker returned zero characters — it deployed but "
                 "does not extract.")
    print(f"[smoke] OK — {data['num_pages']} pages, {chars} chars")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="classic", choices=["classic", "vlm"])
    ap.add_argument("--pdf", default=os.path.join(REPO, "Test Blob File.pdf"))
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        sys.exit(f"no test PDF at {args.pdf} — pass --pdf")

    url = deploy()
    print(f"[deploy] {url}")
    smoke(url, args.pipeline, args.pdf)
    print(f"\nDEPLOYED AND SERVING: {url}\nSet DOCLING_URL={url}/extract")
