import cv2
import numpy as np

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def expand_bbox(x, y, w, h, width, height, margin=0.2):
    extra_w = int(w * margin)
    extra_h = int(h * margin)
    x1 = max(0, x - extra_w)
    y1 = max(0, y - extra_h)
    x2 = min(width, x + w + extra_w)
    y2 = min(height, y + h + extra_h)
    return x1, y1, x2, y2


def center_square_crop(image):
    height, width = image.shape[:2]
    side = min(height, width)
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    return image[y1:y1 + side, x1:x1 + side]


def make_square_with_padding(image):
    height, width = image.shape[:2]
    if height == width:
        return image

    side = max(height, width)
    pad_top = (side - height) // 2
    pad_bottom = side - height - pad_top
    pad_left = (side - width) // 2
    pad_right = side - width - pad_left

    # Replicated padding keeps facial proportions better than stretching the crop to a square.
    return cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )


def crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )

    if len(faces) == 0:
        return center_square_crop(image), False

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    x1, y1, x2, y2 = expand_bbox(x, y, w, h, image.shape[1], image.shape[0])
    return image[y1:y2, x1:x2], True


def resize_square_image(image, size):
    interpolation = cv2.INTER_AREA if max(image.shape[:2]) >= size else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def preprocess_face_bgr(image, output_size):
    cropped, found_face = crop_face(image)
    squared = make_square_with_padding(cropped)
    resized = resize_square_image(squared, output_size)
    return resized, found_face


def preprocess_face_rgb(image_rgb, output_size):
    image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
    processed_bgr, found_face = preprocess_face_bgr(image_bgr, output_size)
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    return processed_rgb, found_face
