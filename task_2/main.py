import torch
import torch.nn as nn
import torch.optim as optim
import requests
import numpy as np
from torchvision import transforms, datasets
from tqdm import tqdm

# Ustawienia API
QUERY_URL = "http://149.156.182.9:6060/task-2/query"
SUBMIT_URL = "http://149.156.182.9:6060/task-2/submit"


# Model kradzionego enkodera (np. ResNet-18)
class StolenEncoder(nn.Module):
    def __init__(self):
        super(StolenEncoder, self).__init__()
        self.encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.encoder.fc = nn.Linear(512, 1024)  # Dostosowanie do rozmiaru wyjścia API

    def forward(self, x):
        return self.encoder(x)


# Funkcja do zadawania zapytań do API
def query_api(images):
    images = images.numpy().tolist()  # Konwersja na format JSON
    response = requests.post(QUERY_URL, json={"images": images})

    if response.status_code != 200:
        print(f"Error: API returned status code {response.status_code}")
        return None

    try:
        data = response.json()
        return torch.tensor(np.array(data['representations']), dtype=torch.float32)
    except (KeyError, ValueError):
        print("Error: Unexpected API response format")
        return None


def train_stolen_model():
    # Przygotowanie danych (np. CIFAR-10)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Inicjalizacja modelu, optymalizatora i straty
    stolen_model = StolenEncoder()
    optimizer = optim.Adam(stolen_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Trening modelu (pętla główna)
    for epoch in range(1):  
        for images, _ in tqdm(dataloader):
            images = images
            with torch.no_grad():
                victim_outputs = query_api(images.cpu())

            # Oblicz przewidywania skradzionego modelu
            stolen_outputs = stolen_model(images)

            # Oblicz stratę
            loss = loss_fn(stolen_outputs, victim_outputs)

            # Aktualizacja wag
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Zapisanie modelu do formatu ONNX
    torch.onnx.export(stolen_model, torch.randn(1, 3, 32, 32), "stolen_encoder.onnx")
    print("Model zapisany jako stolen_encoder.onnx")

    # # Wysłanie do oceny
    # with open("stolen_encoder.onnx", "rb") as f:
    #     response = requests.post(SUBMIT_URL, files={"file": f})
    #     print("Wynik oceny:", response.json())


if __name__ == "__main__":
    train_stolen_model()