from dataclasses import dataclass
import tiktoken
from abc import ABC

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

@dataclass (frozen=True)
class BPETokenizerParams:
    vocab: dict[str, bytes]  # index -> bytes
    merges: list[tuple[str, str]] # index1,index2 -> new_index


class CharacterTokenizer(Tokenizer):
    """Represents a string as a sequence of Unicode code points."""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """Represents a string as a sequence of bytes."""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8") #@ inspect string_bytes
        return list(map(int, string_bytes))  #@ inspect indices

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)
        string_decoded = string_bytes.decode("utf-8")
        return string_decoded



class BytePairEncodingTokenizer(Tokenizer):
    """BPE Tokenizer given a set of merges and a vocab"""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))
        # This is a very slow implementation
        for pair, new_index in self.params.merges:
            indices= merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8")
        return string

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """ Return 'indices' but with all instances of 'pair replaced with new_index"""
    new_indices = []
    i = 0
    while i < len(indices):
        if i+1 < len(indices) and indices[i] == pair[0] and indices[i+1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given 'string' that has been tokenized into 'indices'"""
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens
