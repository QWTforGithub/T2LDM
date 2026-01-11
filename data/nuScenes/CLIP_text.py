#coding=utf-8
from models.CLIP.clip import clip
if __name__ == '__main__':
    text = [
        "There are two cars in the scene.",
        "There are three cars in the scene.",
    ]

    device = "cuda"
    # clip_model = clip.load("ViT-B/32", device=device)
    clip_model = clip.load("ViT-L/14", device=device)
    text_emb = clip.tokenize(text).to(device)
    text_features = clip_model.encode_text(text_emb)

    print(text_features.shape)