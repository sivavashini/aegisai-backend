import os
import gdown

FILE_ID = "1Zh_DbgUiOqCML_d9vkraf9knkaEjSv5o"
DEST = "models/sentinelx_lgbm_calibrated_stage2.pkl"

def download_stage2():
    if os.path.exists(DEST):
        print(f"[startup] {DEST} already exists, skipping download.")
        return
    
    os.makedirs("models", exist_ok=True)
    print(f"[startup] Downloading stage2 model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, DEST, quiet=False)
    print(f"[startup] Download complete.")

if __name__ == "__main__":
    download_stage2()