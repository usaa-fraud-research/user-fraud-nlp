# dashboard_modal.py
import modal
import subprocess

# -----------------------------------------------------------------------------
# 🔧 Modal Image — Includes all dependencies + ML + WordCloud support
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
        "wordcloud",      # wordcloud lib
        "Pillow",         # image rendering backend
        "scikit-learn",   # ✅ needed for LinearSVC / sklearn
        "joblib",         # ✅ save/load sklearn models
    )
    # Copy your app + helper modules + model into the container
    .add_local_file("fraud_dashboard.py", "/root/app.py")
    .add_local_file("semantic_search.py", "/root/semantic_search.py")
    .add_local_file("ml_train.py", "/root/ml_train.py")
    .add_local_dir("models", "/root/models")  # ✅ includes svm_fraud_type.joblib
)

# -----------------------------------------------------------------------------
# 🚀 Modal App Container
# -----------------------------------------------------------------------------
app = modal.App("cfpb-fraud-dashboard", image=image)

# -----------------------------------------------------------------------------
# 🌍 Public Web Deployment
# -----------------------------------------------------------------------------
@app.function(
    secrets=[modal.Secret.from_name("cfpb-dashboard-secret")]  # make sure this secret exists
)
@modal.web_server(port=8000, startup_timeout=600)
def run():
    # Use Popen so Streamlit keeps running after function returns
    subprocess.Popen(
        "streamlit run /root/app.py "
        "--server.port 8000 "
        "--server.address 0.0.0.0 "
        "--server.enableCORS=false "
        "--server.enableXsrfProtection=false",
        shell=True,
    )