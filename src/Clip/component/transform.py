import albumentations as A


def get_transforms(mode="train", config_info = None):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(config_info.size, config_info.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(config_info.size, config_info.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )