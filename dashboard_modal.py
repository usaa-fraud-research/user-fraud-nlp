# dashboard_modal.py
import modal
import subprocess

image = (
    modal.Image.debian_slim(python_version="3.10")
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
    )
    .add_local_file("fraud_dashboard.py", "/root/app.py")
    .add_local_file("semantic_search.py", "/root/semantic_search.py")
)

app = modal.App("cfpb-fraud-dashboard", image=image)

@app.function(
    secrets=[modal.Secret.from_name("cfpb-dashboard-secret")],
)
@modal.web_server(port=8000, startup_timeout=600)   # ⬅ Increased timeout (needs it)
def run():
    # NON-BLOCKING — required for Modal web_server
    subprocess.Popen(
        "streamlit run /root/app.py "
        "--server.port 8000 "
        "--server.address 0.0.0.0 "
        "--server.enableCORS=false "
        "--server.enableXsrfProtection=false",
        shell=True
    )
