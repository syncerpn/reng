import cv2
import numpy as np

def generate_random_image_with_shape(shape):
    return np.random.randint(0, 256, size=shape, dtype=np.uint8)

def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.uint8(np.clip(image, 0.0, 255.0))

# ok
def tobatch(images: list) -> np.ndarray:
    return np.stack(images, axis=0)
    
# ok
def unbatch(tensor: np.ndarray) -> list:
    return [tensor[i,...] for i in range(tensor.shape[0])]

# ok: 4d tensor
def transpose_NHWC_to_NCHW(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return np.transpose(image, (2, 0, 1))
    return np.transpose(image, (0, 3, 1, 2))

# ok: 4d tensor
def transpose_NCHW_to_NHWC(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return np.transpose(image, (1, 2, 0))
    return np.transpose(image, (0, 2, 3, 1))