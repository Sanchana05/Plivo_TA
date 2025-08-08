from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import torch

# load models once (beware memory)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

def caption_image(path: str):
    raw_image = Image.open(path).convert('RGB')
    inputs = processor(images=raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=60)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
