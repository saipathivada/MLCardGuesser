import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import ImageFolder

# Define the model architecture
class CardClassifer(nn.Module):
    def __init__(self, num_classes=53):
        super(CardClassifer, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifer = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        output = self.classifer(x)
        return output

# Load the model
model = CardClassifer(num_classes=53)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()  # Set the model to evaluation mode

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Define the class names
data_dir = '/usr/src/app/card_dataset/train'
class_names = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}

# Function to predict the class of a single image
def predict_image(image_path, model, transform, class_names):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# Example usage
if __name__ == "__main__":
    image_path = '/usr/src/app/card_dataset/test/five of hearts/1.jpg'
    predicted_class = predict_image(image_path, model, transform, class_names)
    print(f'The predicted class is: {predicted_class}')
