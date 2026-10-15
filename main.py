import json
from dataclasses import dataclass

import numpy as np


@dataclass
class TensorMetadata:
    dtype: str
    shape: list
    data_offsets: list


class SafeTensor:
    def __init__(self, model_path):
        self.model_path: str = model_path
        self.model: bytes = self.load()
        self.hdr_size: int = 0
        self.hdr: dict = self.get_hdr()
        self.data: bytes = self.model[self.hdr_size :]

    def load(self) -> bytes:
        try:
            with open(self.model_path, mode="rb") as model:
                return model.read()
        except OSError as err:
            print("OS error:", err)
            return bytes(0)
        except ValueError:
            print("Could not convert data to an integer.")
            return bytes(0)

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

    def get_hdr_data(self, key: str) -> TensorMetadata:
        dtype = self.hdr[key]["dtype"]
        shape = self.hdr[key]["shape"]
        offset = self.hdr[key]["data_offsets"]
        return TensorMetadata(dtype, shape, offset)

    # From BF16 buffer
    def frombuffer(self, name: str) -> np.ndarray:
        metadata: TensorMetadata = self.get_hdr_data(name)
        buf_bf16 = np.frombuffer(
            self.data[metadata.data_offsets[0] : metadata.data_offsets[1]],
            dtype=np.uint16,
        ).astype(np.uint32)
        return np.left_shift(buf_bf16, 16).view(np.float32).reshape(metadata.shape)


def rms_norm(a: np.ndarray, weights: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    msr = np.sqrt(np.mean(np.square(a)) + eps)
    return (a / msr) * weights


def main():
    # 1.Loading
    safetensor = SafeTensor("./SmolLM2-135M/model.safetensors")
    embed = safetensor.frombuffer("model.embed_tokens.weight")
    input_layernorm = safetensor.frombuffer("model.layers.0.input_layernorm.weight")

    # 2. Input
    x = embed[11265]  # Hello token
    x_norm = rms_norm(x, input_layernorm)
    self_attn_q_proj = safetensor.frombuffer("model.layers.0.self_attn.q_proj.weight")
    q = self_attn_q_proj @ x_norm

    # 3. Key Value Projection
    self_attn_k = safetensor.frombuffer("model.layers.0.self_attn.k_proj.weight")
    self_attn_v = safetensor.frombuffer("model.layers.0.self_attn.v_proj.weight")

    k = self_attn_k @ x_norm
    v = self_attn_v @ x_norm

    print(q.shape, k.shape, v.shape)

    # 4. Reshape
    q = q.reshape(9, 64)
    k = k.reshape(3, 64)
    v = v.reshape(3, 64)

    print(q.shape, k.shape, v.shape)

    # 5. RoPe


if __name__ == "__main__":
    main()
