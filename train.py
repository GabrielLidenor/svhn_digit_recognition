import argparse
import config
from src.data_loader import load_and_preprocess_data
from src.models import SVHNCNN, SVHNANN
from src.evaluate import evaluate_predictions

def main(model_type: str):
    print("Step 1: Loading and preprocessing SVHN data...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    print(f"\nStep 2: Initializing {model_type.upper()} Model...")

    # 1. Dynamically select the model class based on user input
    if model_type == "cnn":
        model_builder = SVHNCNN(
            input_shape=config.INPUT_SHAPE,
            num_classes=config.NUM_CLASSES
        )
    elif model_type == "ann":
        model_builder = SVHNANN(
            input_shape=config.INPUT_SHAPE,
            num_classes=config.NUM_CLASSES
        )
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")

    # 2. Build and compile the Keras model
    model = model_builder.build()
    model.summary()

    print("\nStep 3: Beginning Model Training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        verbose=1
    )

    print("\nStep 4: Evaluating Model Performance...")
    evaluate_predictions(model, X_test, y_test)

    # 3. Save the trained model weights/artifacts
    print(f"\nStep 5: Saving Model to disk as svhn_{model_type}.keras...")
    model.save(f"svhn_{model_type}.keras")
    print("Pipeline execution complete! 🚀")

if __name__ == "__main__":
    # 4. Use argparse to allow CLI model selection
    parser = argparse.ArgumentParser(description="SVHN Digit Recognition Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "ann"],
        help="Select the model architecture to train: 'cnn' or 'ann'"
    )

    args = parser.parse_args()
    main(model_type=args.model)