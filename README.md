# ERA-CAPSTONE

## Phi2 : Pretraining LLM from Scratch
### Details
1. Model used: Microsoft Phi2
2. Dataset used: Tiny Stories dataset & Realtime data from finetuned Phi2 model via Ollama
3. Pretraining approach: Pretraining using QLoRA

### Design
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/3a7b2b15-7e70-4ae5-8400-6a4d8dbf5ff9)

### Training Loss Curve
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/f09b0f73-9da2-4bf0-bb00-6e7be8ef8a8e)

### Training Logs
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/a6c143d0-c63c-4227-804f-93a4a8b74f7f)


## Phi2 : Multimodal Finetuning
### Details
1. LLM Backbone: Phi2
2. Vision Tower: clip-vit-large-patch14-336
3. Audio Model: Whisper
4. Pretraining Dataset: LAION-CC-SBU dataset with BLIP captions
5. Finetuning Dataset: Instruct 150k dataset based on COCO

### Pretraining
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/b6c37a95-0a56-4b52-8719-3ff56dc1b703)
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/44d9a11b-b28d-47e1-ba1d-d6dc22ebe748)

![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/76543d98-d9fe-4c1a-ac47-3d06e48053ad)

