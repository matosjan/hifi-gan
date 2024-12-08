import os
import shutil

import gdown

def download():
    gdown.download("https://drive.google.com/file/d/1CZw_7IeBlg3Hk3UtIU1Z2B0xTjZXxjrP/view?usp=sharing")
    os.makedirs("./src/best_model_weights", exist_ok=True)
    shutil.move("model_best_ss.pth", "./src/best_model_weights/model_best.pth")


if __name__ == "__main__":
    download()