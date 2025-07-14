import os
import shutil

def clear_tmp_directory():
    """
    Clears the contents of tmp/transcriptions and tmp/frames
    without deleting the folders themselves.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_folders = [
        os.path.join(project_root, "tmp", "transcriptions"),
        os.path.join(project_root, "tmp", "frames")
    ]

    for folder in target_folders:
        if not os.path.exists(folder):
            print(f"Directory not found: {folder}")
            continue

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
