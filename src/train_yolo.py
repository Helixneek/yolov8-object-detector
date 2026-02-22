import argparse
from ultralytics import YOLO

def train_model(epochs, batch_size):
    # Train on synthetic data
    print(f"Starting training on synthetic data with {epochs} epochs with batch size {batch_size}")

    model = YOLO('yolov8n.pt')
    model.train(data='datasets/syntethic_balls/data.yaml', epochs=epochs, batch_size=batch_size, imgsz=640)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Custom YOLOv8 Model.")

    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50, 
        help="Number of training epochs."
    )
    
    parser.add_argument(
        "--batch", 
        type=int, 
        default=16, 
        help="Batch size for training."
    )

    args = parser.parse_args()

    # Execute the training loop using the terminal inputs
    train_model(epochs=args.epochs, batch_size=args.batch)