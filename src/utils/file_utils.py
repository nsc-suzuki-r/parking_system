import os
import uuid
import glob
import shutil


def clear_existing_files(folder):
    for existing_file in glob.glob(os.path.join(folder, "*")):
        os.remove(existing_file)


def save_file(file, folder):
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(folder, filename)
    file.save(filepath)
    return filename, filepath


def move_image_to_category(folder, img_file, category):
    classify_dest_dir = {
        "bottom": "dataset_bottom/train",
        "rittai_p": "dataset_rittai_p/train",
        "takeda_a": "dataset_takeda_a/train",
        "takeda_b": "dataset_takeda_b/train",
        "takeda_c": "dataset_takeda_c/train",
        "takeda_d": "dataset_takeda_d/train",
    }
    dest_path = os.path.join(classify_dest_dir[folder], category)
    os.makedirs(dest_path, exist_ok=True)
    shutil.move(
        os.path.join("static/split_data", folder, img_file),
        os.path.join(dest_path, img_file),
    )
