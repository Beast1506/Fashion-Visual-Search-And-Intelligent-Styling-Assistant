from torchvision import models, transforms
from PIL import Image
import torch
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove classifier
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_t).squeeze().cpu().numpy()
    return embedding.flatten()

def process_folder(image_folder, output_npy):
    embeddings = []
    filenames = []
    for fname in os.listdir(image_folder):
        fpath = os.path.join(image_folder, fname)
        emb = extract_embedding(fpath)
        embeddings.append(emb)
        filenames.append(fname)
    np.save(output_npy, {'embeddings': np.array(embeddings), 'filenames': filenames})


if __name__ == "__main__":
    print("Extracting embeddings for dresses...")
    process_folder('images/dresses', 'dresses_embeddings.npy')
    print("Extracting embeddings for jeans...")
    process_folder('images/jeans', 'jeans_embeddings.npy')
    print("Done.")