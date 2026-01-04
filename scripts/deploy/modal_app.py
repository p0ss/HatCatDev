"""
Modal deployment for HatCat - complete stack.

Deploys both:
- HatCat GPU backend (Gemma + lens pack + divergence analysis)
- OpenWebUI frontend (chat interface)

Usage:
    modal deploy scripts/deploy/modal_app.py
"""

import modal

app = modal.App("hatcat")

# =============================================================================
# Image: Everything needed for HatCat + OpenWebUI
# =============================================================================

hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "build-essential")
    # Node.js for OpenWebUI frontend
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "git lfs install",
    )
    # Clone repos (shallow clone, faster)
    .run_commands(
        "git clone --depth 1 https://github.com/p0ss/HatCat.git /app",
        "git clone --depth 1 https://github.com/p0ss/HatCat-OpenWebUI.git /ui",
    )
    # Install HatCat deps from requirements.txt
    .run_commands(
        "pip install -r /app/requirements.txt",
    )
    .run_commands(
        "python -c \"import nltk; nltk.download('wordnet')\"",
    )
    # Download lens pack from HuggingFace
    .run_commands(
        "cd /app && git clone https://huggingface.co/HatCatFTW/lens-gemma-3-4b-first-light-v1 lens_packs/gemma-3-4b_first-light-v1-bf16"
    )
    # Download Gemma models at build time (cached in image)
    .env({"HF_HOME": "/models"})
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-4b-pt')\"",
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download('google/gemma-2-2b')\"",
        secrets=[hf_secret],
    )
    # Install OpenWebUI from pyproject.toml + build frontend
    .run_commands(
        "cd /ui && pip install .",
        "cd /ui && npm ci --legacy-peer-deps",
        "cd /ui && npm run build",
    )
    # OpenWebUI config
    .env({
        "ENABLE_SIGNUP": "true",
        "WEBUI_AUTH": "true",
        "DATA_DIR": "/data",
        "HF_HOME": "/models",
    })
)

# Persistent volume for user data only (models baked into image)
data_volume = modal.Volume.from_name("hatcat-data", create_if_missing=True)

# =============================================================================
# Configuration
# =============================================================================

# Users authenticate via OpenWebUI's built-in auth
# First user to sign up becomes admin

# =============================================================================
# Combined Server
# =============================================================================

@app.cls(
    image=image,
    gpu="T4",  # 16GB VRAM
    timeout=600,
    scaledown_window=300,  # Scale to zero after 5 min
    volumes={
        "/data": data_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],  # HF_TOKEN for gated models
)
class HatCat:

    @modal.enter()
    def startup(self):
        import os
        import sys
        import subprocess

        # Configure OpenWebUI before it starts
        os.environ["OPENAI_API_BASE_URL"] = "http://localhost:8765/v1"
        os.environ["OPENAI_API_KEY"] = "sk-local"
        os.environ["DATA_DIR"] = "/data"
        os.environ["DATABASE_URL"] = "sqlite:////data/webui.db"
        os.environ["ENABLE_SIGNUP"] = "true"
        os.environ["WEBUI_AUTH"] = "true"

        # Start HatCat backend on port 8765
        sys.path.insert(0, "/app")
        os.chdir("/app")

        # Start HatCat server in background
        # Explicitly include HF_TOKEN for gated model access
        hatcat_env = {**os.environ, "PYTHONPATH": "/app"}
        if "HF_TOKEN" in os.environ:
            hatcat_env["HF_TOKEN"] = os.environ["HF_TOKEN"]
            hatcat_env["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
            print(f"HF_TOKEN present: {os.environ['HF_TOKEN'][:10]}...")
        else:
            print("WARNING: HF_TOKEN not found in environment!")

        self.hatcat_proc = subprocess.Popen(
            ["python", "-m", "uvicorn", "src.ui.openwebui.server:app",
             "--host", "0.0.0.0", "--port", "8765"],
            cwd="/app",
            env=hatcat_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        print("HatCat backend starting on :8765")

        # Wait for server to be ready (max 60s)
        import time
        import urllib.request
        for i in range(60):
            try:
                urllib.request.urlopen("http://localhost:8765/health", timeout=1)
                print(f"HatCat backend ready after {i+1}s")
                break
            except Exception:
                # Check if process died
                if self.hatcat_proc.poll() is not None:
                    output = self.hatcat_proc.stdout.read().decode() if self.hatcat_proc.stdout else ""
                    print(f"HatCat backend FAILED to start! Exit code: {self.hatcat_proc.returncode}")
                    print(f"Output: {output[:2000]}")
                    raise RuntimeError(f"HatCat failed to start: {output[:500]}")
                time.sleep(1)
        else:
            print("WARNING: HatCat backend didn't respond in 60s, continuing anyway...")

        print("OpenWebUI will connect to local HatCat")

    @modal.exit()
    def shutdown(self):
        if hasattr(self, 'hatcat_proc'):
            self.hatcat_proc.terminate()

    @modal.asgi_app()
    def web(self):
        import sys
        sys.path.insert(0, "/ui/backend")

        # Import OpenWebUI app
        from open_webui.main import app
        return app


# =============================================================================
# Optional: Keep warm
# =============================================================================

# @app.function(schedule=modal.Cron("*/15 8-20 * * *"))
# def keep_warm():
#     import requests
#     requests.get("https://YOUR-URL.modal.run/", timeout=60)


@app.local_entrypoint()
def main():
    print("""
HatCat Modal Deployment (Complete Stack)
=========================================

Deploy:
    modal deploy scripts/deploy/modal_app.py

Access:
    https://YOUR-WORKSPACE--hatcat-hatcat-web.modal.run

First user to sign up becomes admin.
OpenWebUI connects to HatCat backend automatically.

Cost: ~$0.59/hr when active, $0 when idle.
""")
