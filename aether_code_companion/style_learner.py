# aether_code_companion/style_learner.py

import joblib
import logging
from typing import Any, List, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator # For type hinting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StyleModel:
    """
    Manages a pre-trained machine learning model for style prediction.

    This class handles loading a serialized scikit-learn model and using it
    to predict style characteristics based on extracted code features.
    """

    def __init__(self):
        """Initializes the StyleModel with no model loaded."""
        self.model: Optional[BaseEstimator] = None
        self.model_path: Optional[str] = None
        logger.info("StyleModel initialized.")

    def load_model(self, model_path: str) -> bool:
        """
        Loads a pre-trained scikit-learn model from the specified path.

        Args:
            model_path (str): The file path to the serialized model (e.g., .joblib).

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        if not model_path:
            logger.error("Model path cannot be empty.")
            return False

        try:
            self.model = joblib.load(model_path)
            self.model_path = model_path
            logger.info(f"Model successfully loaded from: {model_path}")
            # Basic check to see if the loaded object has a predict method
            if not hasattr(self.model, 'predict'):
                 logger.error(f"Loaded object from {model_path} does not appear to be a valid scikit-learn model (missing predict method).")
                 self.model = None
                 self.model_path = None
                 return False
            return True
        except FileNotFoundError:
            logger.error(f"Model file not found at: {model_path}")
            self.model = None
            self.model_path = None
            return False
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
            self.model = None
            self.model_path = None
            return False

    def predict_style(self, features: Union[List[Any], np.ndarray]) -> Optional[Any]:
        """
        Predicts the style category for a given set of features using the loaded model.

        Args:
            features (Union[List[Any], np.ndarray]): A list or NumPy array
                containing the features extracted from a code snippet, formatted
                as expected by the pre-trained model's predict method (usually
                a 2D array-like structure where each row is a sample).

        Returns:
            Optional[Any]: The prediction result from the model (e.g., a style
                           category label like 'consistent', 'needs_refactoring',
                           or a numerical score). Returns None if the model is
                           not loaded or if prediction fails.
        """
        if self.model is None:
            logger.error("Prediction attempted before loading a model.")
            return None

        try:
            # Ensure features are in the correct format (e.g., 2D array for scikit-learn)
            # If a single sample (1D list/array) is passed, reshape it
            if isinstance(features, list):
                features_np = np.array(features)
            elif isinstance(features, np.ndarray):
                features_np = features
            else:
                logger.error(f"Invalid feature type: {type(features)}. Expected list or numpy array.")
                return None

            if features_np.ndim == 1:
                features_np = features_np.reshape(1, -1)
            elif features_np.ndim != 2:
                 logger.error(f"Invalid feature dimensions: {features_np.ndim}. Expected 1D or 2D array-like.")
                 return None

            prediction = self.model.predict(features_np)
            # If only one prediction was made, return the single result
            if len(prediction) == 1:
                logger.debug(f"Prediction successful for features: {features_np.tolist()}. Result: {prediction[0]}")
                return prediction[0]
            else:
                logger.debug(f"Batch prediction successful. Results: {prediction.tolist()}")
                return prediction # Return the array of predictions for batch input

        except ValueError as ve:
            logger.error(f"Prediction failed due to invalid feature format or values: {ve}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
            return None

# Example Usage (Optional - can be removed or placed under if __name__ == "__main__":)
if __name__ == '__main__':
    # This block demonstrates basic usage.
    # In a real scenario, you'd need a pre-trained model file.
    # Let's assume 'dummy_model.joblib' exists and expects 2 features.

    # Create a dummy model file for demonstration (replace with your actual model)
    try:
        from sklearn.tree import DecisionTreeClassifier
        # Create a simple dummy model that predicts 'consistent' if feature sum > 5, else 'inconsistent'
        X_dummy = np.array([[1, 1], [5, 5], [2, 1], [6, 3]])
        y_dummy = np.array(['inconsistent', 'consistent', 'inconsistent', 'consistent'])
        dummy_model = DecisionTreeClassifier(random_state=42)
        dummy_model.fit(X_dummy, y_dummy)
        joblib.dump(dummy_model, 'dummy_model.joblib')
        logger.info("Created dummy_model.joblib for demonstration.")
    except ImportError:
        logger.warning("scikit-learn not installed. Cannot create dummy model for example.")
        dummy_model = None # Ensure dummy_model exists even if creation fails
    except Exception as e:
        logger.error(f"Error creating dummy model: {e}")
        dummy_model = None

    # --- Actual Example ---
    style_predictor = StyleModel()

    # Attempt to load the model
    model_loaded = style_predictor.load_model('dummy_model.joblib') # Use the dummy model path

    if model_loaded:
        logger.info("Model loaded successfully for example.")

        # Example features (replace with actual features from feature_extractor.py)
        features_consistent = [6, 7]  # Example features likely leading to 'consistent'
        features_inconsistent = [1, 2] # Example features likely leading to 'inconsistent'
        features_batch = [[7, 1], [1, 1], [8, 8]] # Batch prediction example

        # Predict style for single feature sets
        prediction1 = style_predictor.predict_style(features_consistent)
        if prediction1 is not None:
            logger.info(f"Prediction for {features_consistent}: {prediction1}")
        else:
            logger.warning(f"Prediction failed for {features_consistent}")

        prediction2 = style_predictor.predict_style(features_inconsistent)
        if prediction2 is not None:
            logger.info(f"Prediction for {features_inconsistent}: {prediction2}")
        else:
            logger.warning(f"Prediction failed for {features_inconsistent}")

        # Predict style for a batch of features
        batch_predictions = style_predictor.predict_style(features_batch)
        if batch_predictions is not None:
             logger.info(f"Batch predictions for {features_batch}: {batch_predictions}")
        else:
             logger.warning(f"Batch prediction failed for {features_batch}")

        # Example of invalid features
        invalid_features = [1, 2, 3, 4] # Assuming model expects 2 features
        prediction_invalid = style_predictor.predict_style(invalid_features)
        if prediction_invalid is None:
            logger.info("Correctly handled prediction failure for invalid features.")

    else:
        logger.error("Failed to load the model for the example. Ensure 'dummy_model.joblib' exists or provide a valid path.")

    # Clean up the dummy model file
    import os
    if os.path.exists('dummy_model.joblib'):
        try:
            os.remove('dummy_model.joblib')
            logger.info("Cleaned up dummy_model.joblib.")
        except OSError as e:
            logger.warning(f"Could not remove dummy_model.joblib: {e}")