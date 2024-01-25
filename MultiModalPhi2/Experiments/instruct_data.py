from datasets import Dataset, IterableDataset
from PIL import Image

# ChatML format
templates = {
    "assistant": "<|im_start|>assistant\n{msg}<|im_end|>",      # message by assistant
    "user": "<|im_start|>user\n{msg}<|im_end|>"       # message by user
}

ds = Dataset.from_json("llava_instruct_150k.json", split="train")
ds_stream = ds.to_iterable_dataset()


def get_image(image_path):
    image_path = f"train2014/COCO_train2014_{image_path}"
    img = Image.open(image_path)
    return img
    
def get_chatml_text(conversations):
    chatml_text = ""
    for conversation in conversations:
        role = conversation["from"]
        role = "user" if role == "human" else "assistant"
        content = conversation["value"]

        formatted_text = templates[role].format(msg=content)
        chatml_text += formatted_text + "\n"
    return chatml_text

def instruct_data_generator():
    for sample in ds_stream:
        image_path = sample["image"]
        conversations = sample["conversations"]
        
        image = get_image(image_path)
        text = get_chatml_text(conversations)
        yield {"text": text, "image": image}

instruct_ds = IterableDataset.from_generator(generator=instruct_data_generator)