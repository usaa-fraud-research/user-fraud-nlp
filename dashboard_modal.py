# dashboard_modal.py
import modal
import subprocess

# -----------------------------------------------------------------------------
# 🔧 Modal Image — Includes all dependencies + WordCloud support
# -----------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(                    # 👇 Required for WordCloud Image Rendering
        "libfreetype6-dev", 
        "libpng-dev"
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
        "wordcloud",                 # wordcloud lib
        "Pillow"                     # image rendering backend FIX
    )
    # Copy your Streamlit UI + Semantic Search module into container
    .add_local_file("fraud_dashboard.py", "/root/app.py")
    .add_local_file("semantic_search.py", "/root/semantic_search.py")
)

# -----------------------------------------------------------------------------
# 🚀 Modal App Container
# -----------------------------------------------------------------------------
app = modal.App("cfpb-fraud-dashboard", image=image)

# -----------------------------------------------------------------------------
# 🌍 Public Web Deployment
#    — Uses Popen() so Modal doesn't timeout
# -----------------------------------------------------------------------------
@app.function(
    secrets=[modal.Secret.from_name("cfpb-dashboard-secret")]
)
@modal.web_server(port=8000, startup_timeout=600)  # extended load time allowed
def run():
    subprocess.Popen(
        "streamlit run /root/app.py "
        "--server.port 8000 "
        "--server.address 0.0.0.0 "
        "--server.enableCORS=false "
        "--server.enableXsrfProtection=false",
        shell=True
    )
