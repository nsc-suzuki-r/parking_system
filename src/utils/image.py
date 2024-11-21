from PIL import Image
import os


def split_image(filepath, target_folder, models_and_outputs):
    image = Image.open(filepath)
    width, height = image.size
    segment_height = height // len(models_and_outputs)
    segments = []

    for i, (model_path, output_name) in enumerate(models_and_outputs.items()):
        box = (0, i * segment_height, width, (i + 1) * segment_height)
        segment = image.crop(box)
        segment_path = os.path.join(target_folder, output_name)
        segment.save(segment_path)
        segments.append((segment_path, model_path, output_name))

    return segments
