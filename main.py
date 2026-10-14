import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class TensorMetadata:
    dtype: str
    shape: list
    data_offsets: list


class TensorLoader:
    def __init__(self, model_path):
        self.model_path: str = model_path
        self.model: bytes = self.load()
        self.hdr_size: int = 0
        self.hdr: dict = self.get_hdr()
        self.data: bytes = self.model[self.hdr_size :]

    def load(self) -> bytes:
        with open(self.model_path, mode="rb") as model:
            return model.read()

    # Header Size = 30530
    # 00007740  7d 7d 20 20 20 20 20 20  f1 bd e4 3c 45 3d 02 bc  |}}      ...<E=..|
    # Justify the + 8 padding since we have 6 spaces before data (0x20)
    # hexdump -CL -n 30580 SmolLM2-135M/model.safetensors
    def get_hdr(self) -> dict:
        self.hdr_size = int.from_bytes(self.model[0:8], "little") + 8
        return json.loads(self.model[8 : self.hdr_size])

    def save_hdr(self) -> None:
        with open("model.json", mode="w") as file:
            file.write(json.dumps(self.hdr, indent=4))

    def get_header_data(self, key: str) -> TensorMetadata:
        dtype = self.hdr[key]["dtype"]
        shape = self.hdr[key]["shape"]
        offset = self.hdr[key]["data_offsets"]
        return TensorMetadata(dtype, shape, offset)

    def from_bf16_buffer(self, metadata: TensorMetadata) -> ArrayLike:
        buf = np.frombuffer(
            self.data[metadata.data_offsets[0] : metadata.data_offsets[1]],
            dtype=np.uint16,
        ).astype(np.uint32)
        return np.left_shift(buf, 16).view(np.float32).reshape(metadata.shape)

    def get_embed(self) -> ArrayLike:
        return self.from_bf16_buffer(self.get_header_data("model.embed_tokens.weight"))

    def get_input_layernorm(self) -> ArrayLike:
        return self.from_bf16_buffer(
            self.get_header_data("model.layers.0.input_layernorm.weight")
        )


class Tensor:
    def __init__(self, model_path: str):
        self.loader: TensorLoader = TensorLoader(model_path)
        self.eps = 1e-5

    def mean_square_root(self, a: ArrayLike, gamma: ArrayLike) -> ArrayLike:
        msr = np.sqrt((np.sum(np.power(a, 2)) / np.size(a)) + self.eps)
        return np.divide(a, msr) * gamma

    def cosine_sim(self, a: ArrayLike, b: ArrayLike) -> np.float32:
        return np.float32((np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_embed(self):
        return self.loader.get_embed()


def main():
    t = Tensor("./SmolLM2-135M/model.safetensors")
    print(t)


if __name__ == "__main__":
    main()
