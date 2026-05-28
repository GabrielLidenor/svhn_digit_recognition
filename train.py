from src.data_loader import load_and_preprocess_data
from src.models import build_cnn_model  # Swap with build_ann_model if testing ANN version
from src.evaluate import evaluate_predictions
import config

def main():
    print("Step 1: Loading and preprocessing SVHN data...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    print("\nStep 2: Initializing selected Model Version...")
    model = build_cnn_model()
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

if __name__ == "__main__":
    main()
