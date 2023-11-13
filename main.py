from src.data import get_dataset
from src.models.train_model import train_model

batch_size = 32

def main():
    train_loader, valid_loader, test_loader = get_dataset(batch_size)
    train_model(train_loader)
