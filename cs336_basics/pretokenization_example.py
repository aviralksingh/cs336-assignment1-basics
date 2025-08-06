import os
from typing import BinaryIO
import argparse
from tokenizer import ByteTokenizer, CharacterTokenizer, BytePairEncodingTokenizer, BPETokenizerParams

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def main(args):
    bpe_tokenizer = BytePairEncodingTokenizer(BPETokenizerParams(vocab={}, merges=[]))
    with open(args.input_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, args.num_processes, args.split_special_token.encode())
        print(boundaries)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            print(f"Processing chunk from {start} to {end}")
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            tokens= bpe_tokenizer.encode(chunk)
            print(chunk)
            print(tokens)
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunk_tokens = find_chunk_boundaries(f, args.num_processes, args.split_special_token.encode())
            print(f"/----------------------------------------------------/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="The input file to process",required=True)
    parser.add_argument("--num_processes", "-n", default=4, type=int, help="The number of processes to use",required=False)
    parser.add_argument("--split_special_token", "-s", default="<|endoftext|>", type=str, help="The special token to use to split the file",required=False)
    args = parser.parse_args()
    main(args)



    # print(chunk_tokens)

