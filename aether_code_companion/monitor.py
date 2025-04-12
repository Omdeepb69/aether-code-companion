# aether_code_companion/monitor.py

import time
import logging
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, LoggingEventHandler

# Assuming analyzer.py exists in the same directory level or is correctly importable
try:
    from aether_code_companion.analyzer import CodeAnalyzer
except ImportError:
    # Fallback for potential path issues or if analyzer isn't fully defined yet
    # In a real scenario, ensure the project structure allows this import.
    logging.warning("Could not import CodeAnalyzer from aether_code_companion.analyzer. Using a placeholder.")
    # Define a placeholder if needed for testing monitor independently
    class CodeAnalyzer:
        def analyze_file(self, file_path):
            logging.info(f"[Placeholder Analyzer] Analyzing {file_path}")
            # Simulate analysis
            print(f"--- Analyzing {os.path.basename(file_path)} ---")
            # Add placeholder suggestions or analysis results here if desired
            print(f"Placeholder analysis complete for {os.path.basename(file_path)}.")
            print("-" * (len(f"--- Analyzing {os.path.basename(file_path)} ---")))


# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class CodeChangeEventHandler(FileSystemEventHandler):
    """Handles file system events and triggers code analysis for Python files."""

    def __init__(self, analyzer_instance: CodeAnalyzer, watch_patterns=None):
        """
        Initializes the event handler.

        Args:
            analyzer_instance: An instance of the CodeAnalyzer.
            watch_patterns: A list of file extensions to watch (e.g., ['.py']).
                            Defaults to ['.py'].
        """
        super().__init__()
        if not isinstance(analyzer_instance, CodeAnalyzer):
            raise TypeError("analyzer_instance must be an instance of CodeAnalyzer")
        self.analyzer = analyzer_instance
        self.watch_patterns = watch_patterns if watch_patterns else ['.py']
        logging.info(f"CodeChangeEventHandler initialized. Watching patterns: {self.watch_patterns}")

    def _is_watched_file(self, file_path: str) -> bool:
        """Checks if the file path matches the watched patterns."""
        return any(file_path.endswith(ext) for ext in self.watch_patterns)

    def on_modified(self, event):
        """
        Called when a file or directory is modified.

        Args:
            event: The event object representing the file system event.
        """
        if not event.is_directory and self._is_watched_file(event.src_path):
            logging.info(f"Detected modification: {event.src_path}")
            try:
                # Add a small delay to ensure the file write is complete
                time.sleep(0.2)
                self.analyzer.analyze_file(event.src_path)
            except FileNotFoundError:
                logging.error(f"File not found during analysis: {event.src_path}")
            except Exception as e:
                logging.error(f"Error analyzing file {event.src_path}: {e}", exc_info=True)

    def on_created(self, event):
        """
        Called when a file or directory is created.

        Args:
            event: The event object representing the file system event.
        """
        if not event.is_directory and self._is_watched_file(event.src_path):
            logging.info(f"Detected creation: {event.src_path}")
            try:
                # Wait briefly in case it's immediately modified
                time.sleep(0.2)
                self.analyzer.analyze_file(event.src_path)
            except FileNotFoundError:
                 # Might happen if file is created and deleted very quickly
                logging.warning(f"File created but not found shortly after: {event.src_path}")
            except Exception as e:
                logging.error(f"Error analyzing newly created file {event.src_path}: {e}", exc_info=True)

    # Optionally handle on_deleted or on_moved if needed, but for analysis,
    # modification and creation are usually the primary triggers.
    # def on_deleted(self, event):
    #     if not event.is_directory and self._is_watched_file(event.src_path):
    #         logging.info(f"Detected deletion: {event.src_path}")

    # def on_moved(self, event):
    #     if not event.is_directory and self._is_watched_file(event.dest_path):
    #         logging.info(f"Detected move/rename: {event.src_path} to {event.dest_path}")
    #         # Potentially analyze the destination file
    #         try:
    #             time.sleep(0.2)
    #             self.analyzer.analyze_file(event.dest_path)
    #         except FileNotFoundError:
    #             logging.error(f"Moved file not found at destination: {event.dest_path}")
    #         except Exception as e:
    #             logging.error(f"Error analyzing moved file {event.dest_path}: {e}", exc_info=True)


def start_monitoring(path_to_watch: str, analyzer_instance: CodeAnalyzer):
    """
    Starts the file system monitoring process.

    Args:
        path_to_watch: The directory path to monitor recursively.
        analyzer_instance: An instance of the CodeAnalyzer to use for analysis.
    """
    if not os.path.isdir(path_to_watch):
        logging.error(f"Error: Path to watch '{path_to_watch}' is not a valid directory.")
        return

    logging.info(f"Starting file system monitor for path: {path_to_watch}")
    event_handler = CodeChangeEventHandler(analyzer_instance)
    observer = Observer()

    try:
        observer.schedule(event_handler, path_to_watch, recursive=True)
        observer.start()
        logging.info("Observer started. Press Ctrl+C to stop.")
        # Keep the main thread alive until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping observer...")
        observer.stop()
    except Exception as e:
        logging.error(f"An unexpected error occurred in the monitor: {e}", exc_info=True)
        observer.stop() # Attempt to stop observer on other errors too
    finally:
        # Ensure the observer thread is properly joined
        if observer.is_alive():
            observer.join()
        logging.info("Observer stopped and joined. Monitoring terminated.")


if __name__ == "__main__":
    # Example usage:
    # This block allows running monitor.py directly for testing purposes.
    # In a real application, this might be called from a main script.

    print("Aether Code Companion Monitor - Standalone Test")
    print("----------------------------------------------")

    # Create a dummy analyzer instance for testing
    # In the actual project, this would be a properly initialized CodeAnalyzer
    test_analyzer = CodeAnalyzer()

    # Define the path to watch (e.g., the current directory or a specific project)
    # Be cautious when watching large directories.
    watch_path = "." # Watch the current directory by default for testing
    # You might want to change this to a specific test project directory:
    # watch_path = "../path/to/your/test/project"

    if not os.path.isdir(watch_path):
        print(f"Error: Test watch path '{watch_path}' does not exist. Please create it or modify the path.")
    else:
        print(f"Monitoring directory: {os.path.abspath(watch_path)}")
        print("Modify or create Python (.py) files in this directory to trigger analysis.")
        print("Press Ctrl+C to stop.")
        start_monitoring(watch_path, test_analyzer)

    print("Monitor test finished.")