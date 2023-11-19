from torchvision import transforms

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
