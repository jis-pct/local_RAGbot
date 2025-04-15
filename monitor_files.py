from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.opensearch import chunk_and_index_files, delete_documents
import time
import os

class FileChangeHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.pdf')):
            print(f"[+] New file detected: {event.src_path}")
            time.sleep(0.1)
            chunk_and_index_files(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        print(f"[-] File deleted: {event.src_path}")
        time.sleep(0.1)
        delete_documents(os.path.basename(event.src_path))


def monitor_directory(path):
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)  # Keep script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    monitor_directory("uploaded_files")