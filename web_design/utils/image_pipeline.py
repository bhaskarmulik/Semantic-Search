from transformers import CLIPProcessor, CLIPModel
import torch

def get_image_embeddings(self, img) -> list:
    '''
    This function takes in an image and returns the embeddings of the image
    
    Args:
    img: PIL image or image path
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("./clip_model").to(device)

    input = processor(images=img, return_tensors=True).to(device)
    return model.get_image_features(**input)