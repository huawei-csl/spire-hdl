from enum import Enum
import random
from typing import List, Optional, Tuple, Dict

import numpy as np


# enum for


class Encoding(Enum):
    twos_complement = "twos_complement"
    twos_complement_symmetric = "twos_complement_symmetric"
    twos_complement_upper = "twos_complement_upper"
    sign_magnitude = "sign_magnitude"
    sign_magnitude_ext = "sign_magnitude_ext"
    gray = "gray"
    unsigned = "unsigned"
    unsigned_overflow = "unsigned_overflow"
    twos_complement_overflow = "twos_complement_overflow"
    onehot = "onehot"
    sign_magnitude_ext_up = "sign_magnitude_ext_up"

_SIGNED_encodings = {
    Encoding.twos_complement,
    Encoding.twos_complement_symmetric,
    Encoding.sign_magnitude,
    Encoding.sign_magnitude_ext,
    Encoding.sign_magnitude_ext_up,
    Encoding.twos_complement_overflow
}

def is_signed(fmt: Encoding) -> bool:
    return fmt in _SIGNED_encodings

def to_encoding(signed: bool | Encoding) -> Encoding:
    if isinstance(signed, Encoding):
        return signed
    return Encoding.twos_complement if signed else Encoding.unsigned

def from_encoding(fmt: Encoding) -> bool:
    assert fmt in {Encoding.twos_complement, Encoding.unsigned}, f"Cannot convert encoding {fmt} to bool"
    return fmt == Encoding.twos_complement

class EncodingModel:

    def __init__(self, encoding: Encoding):
        self.encoding = encoding

    def value_range(self, width: int) -> Tuple[int, int]:
        fmt = self.encoding
        if fmt in [Encoding.twos_complement, Encoding.sign_magnitude_ext]:
            return (-(1 << (width - 1)), (1 << (width - 1)) - 1)
        if fmt == Encoding.twos_complement_symmetric:
            limit = (1 << (width - 1)) - 1
            return (-limit, limit)
        if fmt == Encoding.twos_complement_overflow:  # signed set membership must match
            return (-(1 << (width - 1)), (1 << (width - 1)) - 1)
        if fmt == Encoding.twos_complement_upper:  # extend on the upper side
            return (-(1 << (width - 1)) + 1, (1 << (width - 1)))
        if fmt in [Encoding.sign_magnitude]:
            limit = (1 << (width - 1)) - 1
            return (-limit, limit)
        if fmt == Encoding.sign_magnitude_ext_up:  # extend on the upper side
            return (-(1 << (width - 1)) + 1, (1 << (width - 1)))  # precedence fix
        if fmt == Encoding.onehot:
            return (0, max(width - 1, 0))
        # unsigned-like encodings (unsigned, gray)
        return (0, (1 << width) - 1)

    @staticmethod
    def _clamp(value: int, lo: int, hi: int) -> int:
        return max(min(value, hi), lo)

    def encode_value(self, value: int, width: int) -> int:
        clamped = self._clamp(value, *self.value_range(width))
        fmt = self.encoding
        if fmt == Encoding.onehot:
            return 1 << clamped
        if fmt == Encoding.gray:
            return clamped ^ (clamped >> 1)
        if fmt in [Encoding.sign_magnitude, Encoding.sign_magnitude_ext]:
            sign_bit = 1 if clamped < 0 else 0
            magnitude = abs(clamped)
            magnitude = magnitude & ((1 << (width - 1)) - 1)  # mask to width-1 bits
            return (sign_bit << (width - 1)) | magnitude
        if fmt == Encoding.sign_magnitude_ext_up:
            sign_bit = 1 if clamped < 0 else 0
            magnitude = abs(clamped)
            magnitude = magnitude & ((1 << (width - 1)) - 1)  # mask to width-1 bits
            if clamped == (1 << (width - 1)):
                sign_bit = 1  # 100..0 represents the upper limit
            return (sign_bit << (width - 1)) | magnitude
        if fmt in [Encoding.twos_complement, Encoding.twos_complement_symmetric, Encoding.twos_complement_upper]:
            if clamped < 0:
                return (1 << width) + clamped  # two's complement representation
            else:
                return clamped
        if fmt == Encoding.unsigned_overflow or fmt == Encoding.twos_complement_overflow:
            return value % (1 << width)
        return clamped

    def get_normal_sample(self, width: int, tb_sigma: float) -> int:
        if tb_sigma is None:
            raise ValueError("tb_sigma must be set to sample normally distributed values")

        lo, hi = self.value_range(width)
        if self.encoding == Encoding.onehot:
            center = (width - 1) / 2 if width > 0 else 0
            raw_value = int(np.round(np.random.normal(center, tb_sigma)))
            raw_value = self._clamp(raw_value, lo, hi)
            return raw_value

        mean = 0 if is_signed(self.encoding) else (lo + hi) / 2
        raw_value = int(np.round(np.random.normal(mean, tb_sigma)))
        raw_value = self._clamp(raw_value, lo, hi)
        return raw_value

    def get_uniform_sample(self, width: int) -> int:
        lo, hi = self.value_range(width)
        if self.encoding == Encoding.onehot:
            if width <= 0:
                return 0
            raw_value = random.randrange(width)
            return raw_value
        raw_value = random.randint(lo, hi)
        return raw_value
    
    def get_normal_sample_np(self, width: int, tb_sigma: float, size: Tuple[int, ...]) -> int:
        out_vals = np.empty(size, dtype=int)
        for idx in np.ndindex(size):
            out_vals[idx] = self.get_normal_sample(width, tb_sigma)
        return out_vals
    
    def get_uniform_sample_np(self, width: int, size: Tuple[int, ...]) -> int:
        out_vals = np.empty(size, dtype=int)
        for idx in np.ndindex(size):
            out_vals[idx] = self.get_uniform_sample(width)
        return out_vals
    

TestVector = Tuple[str, Dict[str, int], Dict[str, int]] # alias for vectors: list of (label, inputs{name->int}, expected{name->int})
TestVectors = List[TestVector]
    
class TestVectorGenerator:
    
    def generate(self) -> TestVectors:
        # output vectors: list of (label, inputs{name->int}, expected{name->int})
        raise NotImplementedError("Subclasses must implement generate() method to produce test vectors")


class TwoInputArithmeticTestVectorsBase(TestVectorGenerator):

    def __init__(
        self,
        a_w: int,
        b_w: int,
        y_w: Optional[int] = None,
        num_vectors: Optional[int] = None,
        tb_sigma: Optional[float] = None,
        a_encoding: Encoding = Encoding.unsigned,
        b_encoding: Encoding = Encoding.unsigned,
        y_encoding: Encoding = Encoding.unsigned,
        seed: Optional[int] = 42,
    ):
        # seed: reseeds the module-global RNGs at generate() time so regressions are
        # reproducible (the FP generators already default to seed=42; the integer ones
        # sampled the unseeded globals — ). Pass None for nondeterministic runs.
        self.seed = seed
        self.a_w = a_w
        self.b_w = b_w
        self.y_w = a_w + b_w if y_w is None else y_w
        self.num_vectors = num_vectors
        self.tb_sigma = tb_sigma
        self.a_encoding_model = EncodingModel(a_encoding)
        self.b_encoding_model = EncodingModel(b_encoding)
        self.y_encoding_model = EncodingModel(y_encoding)
        
    def _reseed(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def generate(self) -> TestVectors:
        raise NotImplementedError()

class MultiplierTestVectors(TwoInputArithmeticTestVectorsBase):

    def generate(self) -> TestVectors:
        self._reseed()

        if self.num_vectors is None:
            self.num_vectors = 64  # default number of vectors

        vecs = []
        for _ in range(self.num_vectors):
            if self.tb_sigma is not None:
                va_value = self.a_encoding_model.get_normal_sample(self.a_w, self.tb_sigma)
                vb_value = self.b_encoding_model.get_normal_sample(self.b_w, self.tb_sigma)
            else:
                va_value = self.a_encoding_model.get_uniform_sample(self.a_w)
                vb_value = self.b_encoding_model.get_uniform_sample(self.b_w)
            va_encoded = self.a_encoding_model.encode_value(va_value, self.a_w)
            vb_encoded = self.b_encoding_model.encode_value(vb_value, self.b_w)

            y_value = va_value * vb_value
            y_encoded = self.y_encoding_model.encode_value(y_value, self.y_w)

            # append test vector
            vecs.append(
                (f"{va_value}*{vb_value}", {"a": va_encoded, "b": vb_encoded}, {"y": y_encoded})
            )

        return vecs


class MultiplierTestVectorsExhaustive(TwoInputArithmeticTestVectorsBase):

    def generate_encoding_tables(self) -> Tuple[dict, dict, dict]:
        a_table = {}
        b_table = {}
        y_table = {}

        a_lo, a_hi = self.a_encoding_model.value_range(self.a_w)
        for val in range(a_lo, a_hi + 1):
            enc = self.a_encoding_model.encode_value(val, self.a_w)
            a_table[enc] = val

        b_lo, b_hi = self.b_encoding_model.value_range(self.b_w)
        for val in range(b_lo, b_hi + 1):
            enc = self.b_encoding_model.encode_value(val, self.b_w)
            b_table[enc] = val

        y_lo, y_hi = self.y_encoding_model.value_range(self.y_w)
        for val in range(y_lo, y_hi + 1):
            enc = self.y_encoding_model.encode_value(val, self.y_w)
            y_table[enc] = val

        return a_table, b_table, y_table

    def generate(self) -> TestVectors:

        if self.num_vectors is not None or self.tb_sigma is not None:
            raise ValueError("num_vectors must be None for exhaustive test vector generation")       

        total_combinations = (1 << self.a_w) * (1 << self.b_w)
        self.num_vectors = total_combinations
        a_table, b_table, y_table = self.generate_encoding_tables()

        vecs = []
        for i in range(self.num_vectors):

            # get encoded value
            va_encoded = i % (1 << self.a_w)
            vb_encoded = i // (1 << self.a_w)

            # Non-surjective encodings (sign_magnitude -0, symmetric most-negative, onehot/gray
            # gaps) have patterns with no value: skip them instead of KeyErroring.
            if va_encoded not in a_table or vb_encoded not in b_table:
                continue
            va_value = a_table[va_encoded]
            vb_value = b_table[vb_encoded]
            y_value = va_value * vb_value
            y_encoded = self.y_encoding_model.encode_value(y_value, self.y_w)  # y_table maps encoded->value

            # append test vector
            vecs.append(
                (f"{va_value}*{vb_value}", {"a": va_encoded, "b": vb_encoded}, {"y": y_encoded})
            )

        return vecs


class AdderTestVectors(TwoInputArithmeticTestVectorsBase):

    def __init__(
        self,
        a_w: int,
        b_w: int,
        y_w: Optional[int] = None,
        num_vectors: Optional[int] = None,
        tb_sigma: Optional[float] = None,
        a_encoding: Encoding = Encoding.unsigned,
        b_encoding: Encoding = Encoding.unsigned,
        y_encoding: Encoding = Encoding.unsigned,
    ):
        y_width = max(a_w, b_w) + 1 if y_w is None else y_w
        super().__init__(
            a_w=a_w,
            b_w=b_w,
            y_w=y_width,
            num_vectors=num_vectors,
            tb_sigma=tb_sigma,
            a_encoding=a_encoding,
            b_encoding=b_encoding,
            y_encoding=y_encoding,
        )

    def generate(self) -> TestVectors:
        self._reseed()
        if self.num_vectors is None:
            self.num_vectors = 64

        vecs = []
        for _ in range(self.num_vectors):
            if self.tb_sigma is not None:
                va_value = self.a_encoding_model.get_normal_sample(self.a_w, self.tb_sigma)
                vb_value = self.b_encoding_model.get_normal_sample(self.b_w, self.tb_sigma)
            else:
                va_value = self.a_encoding_model.get_uniform_sample(self.a_w)
                vb_value = self.b_encoding_model.get_uniform_sample(self.b_w)

            va_encoded = self.a_encoding_model.encode_value(va_value, self.a_w)
            vb_encoded = self.b_encoding_model.encode_value(vb_value, self.b_w)

            y_value = va_value + vb_value
            y_encoded = self.y_encoding_model.encode_value(y_value, self.y_w)

            vecs.append(
                (f"{va_value}+{vb_value}", {"a": va_encoded, "b": vb_encoded}, {"y": y_encoded})
            )

        return vecs


class SubtractorTestVectors(TwoInputArithmeticTestVectorsBase):

    def __init__(
        self,
        a_w: int,
        b_w: int,
        y_w: Optional[int] = None,
        num_vectors: Optional[int] = None,
        tb_sigma: Optional[float] = None,
        a_encoding: Encoding = Encoding.unsigned,
        b_encoding: Encoding = Encoding.unsigned,
        y_encoding: Encoding = Encoding.unsigned,
    ):
        y_width = max(a_w, b_w) + 1 if y_w is None else y_w
        super().__init__(
            a_w=a_w,
            b_w=b_w,
            y_w=y_width,
            num_vectors=num_vectors,
            tb_sigma=tb_sigma,
            a_encoding=a_encoding,
            b_encoding=b_encoding,
            y_encoding=y_encoding,
        )

    def generate(self) -> TestVectors:
        self._reseed()
        if self.num_vectors is None:
            self.num_vectors = 64

        vecs = []
        for _ in range(self.num_vectors):
            if self.tb_sigma is not None:
                va_value = self.a_encoding_model.get_normal_sample(self.a_w, self.tb_sigma)
                vb_value = self.b_encoding_model.get_normal_sample(self.b_w, self.tb_sigma)
            else:
                va_value = self.a_encoding_model.get_uniform_sample(self.a_w)
                vb_value = self.b_encoding_model.get_uniform_sample(self.b_w)

            va_encoded = self.a_encoding_model.encode_value(va_value, self.a_w)
            vb_encoded = self.b_encoding_model.encode_value(vb_value, self.b_w)

            y_value = va_value - vb_value
            y_encoded = self.y_encoding_model.encode_value(y_value, self.y_w)

            vecs.append(
                (f"{va_value}-{vb_value}", {"a": va_encoded, "b": vb_encoded}, {"y": y_encoded})
            )

        return vecs


class EncoderDecoderTestVectors:
    def __init__(
        self,
        width: int,
        num_vectors: Optional[int] = None,
        tb_sigma: Optional[float] = None,
        input_encoding: Encoding = Encoding.twos_complement,
        output_encoding: Encoding = Encoding.sign_magnitude,
        seed: Optional[int] = 42,
    ):
        self.seed = seed
        self.width = width
        self.num_vectors = num_vectors
        self.tb_sigma = tb_sigma
        self.input_encoding_model = EncodingModel(input_encoding)
        self.output_encoding_model = EncodingModel(output_encoding)

    def _reseed(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def _sample_value(self) -> int:
        if self.tb_sigma is not None:
            return self.input_encoding_model.get_normal_sample(self.width, self.tb_sigma)
        return self.input_encoding_model.get_uniform_sample(self.width)

    def generate(self) -> TestVectors:
        self._reseed()
        if self.num_vectors is None:
            self.num_vectors = 64

        vecs = []
        for _ in range(self.num_vectors):
            value = self._sample_value()
            i_encoded = self.input_encoding_model.encode_value(value, self.width)
            o_encoded = self.output_encoding_model.encode_value(value, self.width)
            vecs.append((f"{value}", {"i": i_encoded}, {"o": o_encoded}))

        return vecs
