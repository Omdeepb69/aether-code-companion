# aether_code_companion/main.py

import argparse
import logging
import sys
import time
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler # Placeholder, replace with custom handler

# Attempt to import project modules
try:
    from aether_code_companion.monitor import CodeChangeHandler
    from aether_code_companion.analyzer import Analyzer
    from aether_code_companion.style_learner import StyleLearner
except ImportError as e:
    # Provide helpful message if modules are not found, possibly due to execution context
    print(f"Error importing project modules: {e}")
    print("Ensure you are running this script from the root directory of the project or have installed the package.")
    # Add project root to sys.path if running script directly for development
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to sys.path")
        # Retry imports
        from aether_code_companion.monitor import CodeChangeHandler
        from aether_code_companion.analyzer import Analyzer
        from aether_code_companion.style_learner import StyleLearner


# --- Configuration ---
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_CONFIG_PATH = "config/rules.json" # Example path
DEFAULT_MODEL_PATH = "models/style_model.pkl" # Example path
DEFAULT_LEARNING_DATA_DIR = "learning_data" # Directory for style learner examples

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Main Application Logic ---

def main():
    """
    Main entry point for the Aether Code Companion.
    Initializes components and starts the file monitoring process.
    """
    parser = argparse.ArgumentParser(
        description="Aether Code Companion: An AI agent for code analysis and suggestions."
    )
    parser.add_argument(
        "watch_dir",
        type=str,
        help="The directory path to monitor for code changes.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the analysis rules configuration file (JSON). Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to load/save the style learner model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--learn-dir",
        type=str,
        default=DEFAULT_LEARNING_DATA_DIR,
        help=f"Directory containing code examples for initial style learning. Default: {DEFAULT_LEARNING_DATA_DIR}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Monitor the directory recursively.",
    )

    args = parser.parse_args()

    watch_path = Path(args.watch_dir)
    config_path = Path(args.config)
    model_path = Path(args.model)
    learn_dir_path = Path(args.learn_dir)

    # --- Validate Paths ---
    if not watch_path.is_dir():
        logger.error(f"Error: Watch directory not found or is not a directory: {watch_path}")
        sys.exit(1)

    # Ensure parent directories for config and model exist if specified paths are not absolute
    # (This is more about saving than loading, but good practice)
    if not config_path.is_absolute() and not config_path.parent.exists():
        logger.warning(f"Config directory {config_path.parent} does not exist. Attempting to create.")
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create config directory {config_path.parent}: {e}")
            # Decide if this is fatal or not; maybe Analyzer handles missing config gracefully
            # sys.exit(1)

    if not model_path.is_absolute() and not model_path.parent.exists():
        logger.warning(f"Model directory {model_path.parent} does not exist. Attempting to create.")
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create model directory {model_path.parent}: {e}")
            # Decide if this is fatal or not
            # sys.exit(1)

    # --- Initialize Components ---
    try:
        logger.info("Initializing Analyzer...")
        analyzer = Analyzer(config_path=config_path) # Analyzer handles config loading internally
        logger.info("Analyzer initialized.")

        logger.info("Initializing StyleLearner...")
        style_learner = StyleLearner(model_path=model_path)
        # Perform initial training/loading if learning data exists
        if learn_dir_path.is_dir():
            logger.info(f"Found learning data directory: {learn_dir_path}. Training/updating style model...")
            style_learner.learn_from_directory(learn_dir_path)
            logger.info("Style model training/update complete.")
        elif model_path.exists():
             logger.info(f"Loading existing style model from: {model_path}")
             # Loading is typically handled within StyleLearner's __init__ or a load method
        else:
            logger.warning(f"No learning data directory ({learn_dir_path}) or existing model ({model_path}) found. Style learner will start fresh.")
        logger.info("StyleLearner initialized.")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        sys.exit(1)

    # --- Setup File Monitoring ---
    logger.info(f"Setting up file monitor for directory: {watch_path}")
    event_handler = CodeChangeHandler(analyzer=analyzer, style_learner=style_learner)
    observer = Observer()
    observer.schedule(event_handler, str(watch_path), recursive=args.recursive)

    # --- Start Monitoring ---
    logger.info("Starting Aether Code Companion...")
    observer.start()
    logger.info(f"Monitoring {'recursively' if args.recursive else 'non-recursively'} in '{watch_path}'. Press Ctrl+C to stop.")

    try:
        while True:
            # Keep the main thread alive while the observer runs in a background thread
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received (Ctrl+C).")
    except Exception as e:
        logger.error(f"An unexpected error occurred during monitoring: {e}", exc_info=True)
    finally:
        logger.info("Stopping file monitor...")
        observer.stop()
        observer.join() # Wait for the observer thread to finish
        logger.info("Aether Code Companion stopped.")
        # Optionally save the learned style model on exit
        try:
            logger.info(f"Saving style model to {model_path}...")
            style_learner.save_model()
            logger.info("Style model saved.")
        except Exception as e:
            logger.error(f"Failed to save style model: {e}", exc_info=True)


if __name__ == "__main__":
    main()