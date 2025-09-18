from data import load_and_preprocess_data
from model import NCF
from train import train_model
from evaluate import evaluate_model

# Config
file_path = 'ecommerce_data.csv'  # Thay bằng path thật
implicit = True  # Hoặc False cho explicit

train_loader, val_loader, test_loader, num_users, num_items = load_and_preprocess_data(file_path, implicit=implicit)

model = NCF(num_users, num_items)
model = train_model(model, train_loader, val_loader)

metrics = evaluate_model(model, test_loader, K=5)
print(metrics)