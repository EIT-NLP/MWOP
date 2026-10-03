from typing import Any, Mapping, Optional

def get_image_files_from_sample(sample: Mapping[str, Any]) -> Optional[Any]:
    if 'image' in sample:
        image_files = sample['image']
    elif 'images' in sample:
        image_files = sample['images']
    else:
        return None
    if isinstance(image_files, list) and len(image_files) == 0:
        return None
    return image_files

def sample_has_image(sample: Mapping[str, Any]) -> bool:
    return get_image_files_from_sample(sample) is not None
