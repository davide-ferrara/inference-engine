import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class TensorData:
    dtype: str
    shape: list
    data_offsets: list


@dataclass
class Header:
    embed_tokens: TensorData


class Model:
    def __init__(self):
        self.header: Header
        self.embed_weights = None
        self.eps = 1e-5


def mean_square_root(a: ArrayLike, gamma: ArrayLike) -> ArrayLike:
    msr = np.sqrt((np.sum(np.power(a, 2)) / np.size(a)) + 1e-5)
    return np.divide(a, msr) * gamma


def cosine_sim(a: ArrayLike, b: ArrayLike) -> np.float32:
    return np.float32((np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_weights(filename):
    print(f'Loaded "{filename}"')
    with open(filename, mode="rb") as file:
        return file.read()


def get_header(model, size) -> dict:
    return json.loads(model[8:size])


def save_header(header):
    with open("model.json", mode="w") as file:
        file.write(json.dumps(header, indent=4))


def get_embed(data, header):
    print(header["model.embed_tokens.weight"])
    shape = header["model.embed_tokens.weight"]["shape"]
    offset = header["model.embed_tokens.weight"]["data_offsets"][1]
    embed_bf16 = np.frombuffer(data[:offset], dtype=np.uint16).astype(np.uint32)
    embed = np.left_shift(embed_bf16, 16).view(np.float32).reshape(shape)
    return embed


def get_layer_norm(data, header):
    print(header["model.layers.0.input_layernorm.weight"])
    offset = header["model.layers.0.input_layernorm.weight"]["data_offsets"]
    shape = header["model.layers.0.input_layernorm.weight"]["shape"]
    ln_bf16 = np.frombuffer(data[offset[0] : offset[1]], dtype=np.uint16).astype(
        np.uint32
    )
    ln = np.left_shift(ln_bf16, 16).view(np.float32).reshape(shape)
    return ln


def main():
    model = load_weights("./SmolLM2-135M/model.safetensors")

    # Header Size = 30530
    # 00007740  7d 7d 20 20 20 20 20 20  f1 bd e4 3c 45 3d 02 bc  |}}      ...<E=..|
    # Justify the + 8 padding since we have 6 spaces before data (0x20)
    # hexdump -CL -n 30580 SmolLM2-135M/model.safetensors
    size = int.from_bytes(model[0:8], "little") + 8

    header = get_header(model, size)
    save_header(header)

    # Starts from 0xf1 byte
    data = model[size:]

    embed = get_embed(data, header)
    print(embed.shape)

    # "Hello": 1955 "World": 11265
    hello = embed[19556]
    print("Cosine similarity:", cosine_sim(hello, embed[11265]))
    print("Cosine similarity:", cosine_sim(hello, embed[19556]))
    print("Cosine similarity:", cosine_sim(hello, -embed[19556]))

    ln = get_layer_norm(data, header)
    print("LayerNorm:", ln)

    tensor = mean_square_root(hello, ln)
    print("MeanSquareRes:", tensor)


if __name__ == "__main__":
    main()
