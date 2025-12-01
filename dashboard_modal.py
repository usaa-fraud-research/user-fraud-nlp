# dashboard_modal.py
import os
import subprocess
import modal

# -----------------------------------------------------------------------------
# 🔧 Modal Image — Includes all dependencies + WordCloud + joblib + model file
# -----------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(                    # For WordCloud image rendering
        "libfreetype6-dev",
        "libpng-dev",
    )
    .pip_install(
        "streamlit",
        "supabase",
        "pandas",
        "plotly",
        "python-dotenv",
        "openai",
        "numpy",
        "tqdm",
        "beautifulsoup4",
        "dateparser",
        "wordcloud",
        "Pillow",
        "joblib",                    # ✅ needed for loading the ML model
    )
    # Copy your app + helper modules into the container
    .add_local_file("fraud_dashboard.py", "/root/app.py")
    .add_local_file("semantic_search.py", "/root/semantic_search.py")
    .add_local_file("ml_train.py", "/root/ml_train.py")   # ✅ so `import ml_train` works
    # Copy the models directory so relative path "models/..." still works
    .add_local_dir("models", "/root/models")
)

# -----------------------------------------------------------------------------
# 🚀 Modal App
# -----------------------------------------------------------------------------
app = modal.App("cfpb-fraud-dashboard", image=image)

# -----------------------------------------------------------------------------
# 🌍 Public Web Deployment
# -----------------------------------------------------------------------------
@app.function(
    # secret should contain SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY
    secrets=[modal.Secret.from_name("custom-secret")]
)
@modal.web_server(port=8000, startup_timeout=600)
def run():
    # Make sure imports & relative paths use /root
    os.chdir("/root")

    subprocess.Popen(
        [
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            "8000",
            "--server.address",
            "0.0.0.0",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
        ]
    )