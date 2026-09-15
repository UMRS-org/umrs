#!/usr/bin/python3
import requests

headers = {"Authorization": f"Bearer {api_key}"}

# Get list of models from NGC
response = requests.get("https://api.ngc.nvidia.com/v2/models", headers=headers)

if response.status_code == 200:
    data = response.json()
    print("✅ Connected to NGC\n")

    # Directly access the "models" list
    for model in data.get("models", []):
        print(f"🔹 Name: {model.get('name')}")
        print(f"   📁 Display Name: {model.get('displayName')}")
        print(f"   🏷️ Labels: {', '.join(model.get('labels', []))}")
        print(f"   🛠 Framework: {model.get('framework')}")
        print(f"   📦 Format: {model.get('modelFormat')}")
        print(f"   ⚖️ Precision: {model.get('precision')}")
        print(f"   📅 Updated: {model.get('updatedDate')}")
        print(f"   📝 Description: {model.get('shortDescription')}")
        print(f"   🔗 Model ID: {model.get('name')}")
        print("-" * 60)

else:
    print("❌ Error:", response.status_code, response.text)
