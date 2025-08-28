#!/usr/bin/python3

import requests

api_key = open("NGC_API_Key.txt", "r").read().strip()

headers = {"Authorization": f"Bearer {api_key}"}
model_id = "resnet_50_pyt_ckpt"

url = f"https://api.ngc.nvidia.com/v2/models/nvidia/{model_id}/versions/latest/files"
print(url)

response = requests.get(url, headers=headers)

if response.status_code == 200:
    with open("resnet_50_pyt_ckpt.zip", "wb") as f:
        f.write(response.content)
    print("✅ ResNet50 downloaded successfully.")
else:
    print("❌ Error:", response.status_code, response.text)
