import os
import uuid
import glob


def clear_existing_files(folder):
    for existing_file in glob.glob(os.path.join(folder, "*")):
        os.remove(existing_file)


def save_file(file, folder):
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(folder, filename)
    file.save(filepath)
    return filename, filepath
