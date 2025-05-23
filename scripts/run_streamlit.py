import os
import subprocess
import webbrowser
import threading
import time


def launch_browser():
    """Wait briefly and open the browser."""
    time.sleep(2)
    webbrowser.open("http://localhost:8501")


if __name__ == "__main__":
    app_path = os.path.abspath("scripts/app.py")

    if not os.path.exists(app_path):
        print("❌ app.py not found at:", app_path)
        exit(1)

    threading.Thread(target=launch_browser).start()

    try:
        subprocess.run(["streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError as e:
        print("❌ Streamlit failed:", e)
