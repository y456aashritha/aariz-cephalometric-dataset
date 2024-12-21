import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from dataLoader import Rescale, ToTensor, LandmarksDataset
import models
import train
import lossFunction

plt.ion()  # Interactive mode

# Visualization Function
def visualize_landmarks(image, ground_truth, predicted, image_name=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    
    for (x, y) in ground_truth:
        plt.scatter(x, y, c='red', s=30, label='Ground Truth')
    
    for (x, y) in predicted:
        plt.scatter(x, y, c='blue', s=30, marker='x', label='Predicted')

    if image_name:
        plt.title(f"Landmark Visualization: {image_name}")
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='upper right')
    plt.show()

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int, default=1)
    parser.add_argument("--landmarkNum", type=int, default=19)
    parser.add_argument("--image_scale", default=(800, 640), type=tuple)
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--dataRoot", type=str, default="process_data/")
    parser.add_argument("--trainingSetCsv", type=str, default="cepha_train.csv")
    parser.add_argument("--testSetCsv", type=str, default="cepha_val.csv")
    return parser.parse_args()

# Main Training Function
def main():
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")

    model_ft = models.fusionVGG19(torchvision.models.vgg19_bn(pretrained=True), config).to(device)

    print("Image scale:", config.image_scale)
    print("Using GPU:", config.use_gpu)

    transform = torchvision.transforms.Compose([
        Rescale(config.image_scale),
        ToTensor()
    ])

    train_dataset = LandmarksDataset(
        csv_file=config.dataRoot + config.trainingSetCsv,
        root_dir=config.dataRoot + 'cepha/',
        transform=transform,
        landmarksNum=config.landmarkNum
    )

    val_dataset = LandmarksDataset(
        csv_file=config.dataRoot + config.testSetCsv,
        root_dir=config.dataRoot + 'cepha/',
        transform=transform,
        landmarksNum=config.landmarkNum
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.batchSize, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batchSize, shuffle=False, num_workers=8)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    criterion = lossFunction.fusionLossFunc_improved(config)
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1.0)

    train.train_model(model_ft, dataloaders, criterion, optimizer, config)

if __name__ == "__main__":
    main()
