from diffusers import DiffusionPipeline
import torch
import cv2
import numpy as np
from PIL import Image

PROMPT = "Cute cat"
SEED = 123
STEPS = 20

samples_dir = "samples"
image_width = 256
image_height = 192
model = "shadowlamer/sd-zxspectrum-model-256"


def quantize_to_palette(_image, _palette):
    x_query = _image.reshape(-1, 3).astype(np.float32)
    x_index = _palette.astype(np.float32)
    knn = cv2.ml.KNearest_create()
    knn.train(x_index, cv2.ml.ROW_SAMPLE, np.arange(len(_palette)))
    ret, results, neighbours, dist = knn.findNearest(x_query, 1)
    _quantized_image = np.array([_palette[idx] for idx in neighbours.astype(int)])
    _quantized_image = _quantized_image.reshape(_image.shape)
    return Image.fromarray(cv2.cvtColor(np.array(_quantized_image, dtype=np.uint8), cv2.COLOR_BGR2RGB))


def collect_char_colors(image, _x, _y):
    _colors = {}
    for _char_y in range(0, 8):
        for _char_x in range(0, 8):
            _pixel = image.getpixel((_x + _char_x, _y + _char_y))
            _colors[_pixel] = 1 if _pixel not in _colors else _colors[_pixel] + 1
    _colors = sorted(_colors.items(), key=lambda _v: _v[1], reverse=True)
    return [list(_tuple[0]) for _tuple in list(_colors)]


def palette_to_attr(_palette):
    if len(_palette) == 0:
        return 0x00
    _attr = 0x40
    _paper = palette[0]
    if _paper[0] > 0:
        _attr = _attr + 0x10  # r
    if _paper[1] > 0:
        _attr = _attr + 0x20  # g
    if _paper[2] > 0:
        _attr = _attr + 0x08  # b
    if len(palette) == 1:
        return _attr
    _ink = palette[1]
    if _ink[0] > 0:
        _attr = _attr + 0x02  # r
    if _ink[1] > 0:
        _attr = _attr + 0x04  # g
    if _ink[2] > 0:
        _attr = _attr + 0x01  # b
    return _attr


pipe = DiffusionPipeline.from_pretrained(model, safety_checker=None, requires_safety_checker=False)
generator = torch.Generator("cpu").manual_seed(SEED)
raw_image = pipe(PROMPT, height=192, width=256, num_inference_steps=STEPS, generator=generator).images[0]
palette = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 0, 0], [255, 0, 255], [255, 255, 0], [255, 255, 255]])
input_image = np.array(raw_image)
input_image = input_image[:, :, ::-1].copy()
image = quantize_to_palette(_image=input_image, _palette=palette)

byte_buffer = [0] * 0x1800
attr_buffer = [0b00111000] * 0x300

for y in range(0, image_height, 8):
    for x in range(0, image_width, 8):
        px = int(x / 8)
        py = int(y / 8)
        palette = collect_char_colors(image, x, y)
        byte_index = int(py / 8) * 0x800 + (py % 8) * 32 + px
        for cy in range(8):
            byte = 0
            for cx in range(8):
                byte = byte * 2
                pixel = list(image.getpixel((x + cx, y + cy)))
                if palette[0] != pixel:
                    byte = byte + 1
            byte_buffer[byte_index] = byte
            byte_index = byte_index + 0x100
        attr = palette_to_attr(palette)
        attr_buffer[py * 32 + px] = attr

out = samples_dir + "/" + PROMPT.replace(" ", "_") + "_" + str(SEED) + "_" + str(STEPS)

scr = open(out + ".scr", 'wb')
scr.write(bytearray(byte_buffer))
scr.write(bytearray(attr_buffer))
scr.close()

image.save(out + ".png")

