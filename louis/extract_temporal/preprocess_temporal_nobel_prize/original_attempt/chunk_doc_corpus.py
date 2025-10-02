"""Chunk documents from a Document Corpus JSONL file.

This script is used to chunk documents from a Document Corpus JSONL file,
via Chonkie.
"""

import argparse
import json

import chonkie
import msgspec
from tqdm.auto import tqdm

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()


def read_json(file_path, jsonl=False):
    with open(file_path, "r") as file:
        data = file.read()
    if jsonl:
        output = decoder.decode_lines(data)
    else:
        output = decoder.decode(data)

    print(f"The file is of type: {type(output)}")
    print(f"The file contains {len(output)} items.")
    return output


def write_json(file_path, data, jsonl=False):
    with open(file_path, "wb") as file:
        if jsonl:
            file.write(encoder.encode_lines(data))
        else:
            file.write(encoder.encode(data))
    print(f"The file contains {len(data)} items.")
    print("Saved to", file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk documents from a JSONL file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--chunk_by", default="token", choices=["token", "word", "sentence", "recursive"], help="Chunking method to use"
    )
    parser.add_argument("--chunk_size", default=512, type=int, help="Size of chunks")
    parser.add_argument("--tokenizer_name_or_path", default="o200k_base", type=str)

    args = parser.parse_args()

    # Load documents
    print("Loading documents...")
    documents = read_json(args.input_path, jsonl=True)

    # Initialize chunker
    if args.chunk_by == "token":
        chunker = chonkie.TokenChunker(tokenizer=args.tokenizer_name_or_path, chunk_size=args.chunk_size)
    elif args.chunk_by == "word":
        chunker = chonkie.TokenChunker(tokenizer="word", chunk_size=args.chunk_size)
    elif args.chunk_by == "sentence":
        chunker = chonkie.SentenceChunker(tokenizer_or_token_counter=args.tokenizer_name_or_path, chunk_size=args.chunk_size)
    elif args.chunk_by == "recursive":
        chunker = chonkie.RecursiveChunker(
            tokenizer_or_token_counter=args.tokenizer_name_or_path, chunk_size=args.chunk_size, min_characters_per_chunk=1
        )
    else:
        raise ValueError(f"Invalid chunking method: {args.chunk_by}")

    # Process and chunk documents
    print("Chunking documents...")
    chunked_documents = []
    current_chunk_id = 0
    for doc in tqdm(documents):
        text = doc["text"] # .split("\n", 1)
        chunks = chunker.chunk(text)
        for chunk in chunks:
            chunked_doc = {
                "chunkid": current_chunk_id,
                "docid": doc["docid"],
                # "title": title,
                "text": chunk.text,
            }
            chunked_documents.append(chunked_doc)
            current_chunk_id += 1

    # Save chunked documents
    print("Saving chunked documents...")
    write_json(
        args.output_path,
        chunked_documents,
        jsonl=True
    )
    print(f"Done! Processed {len(documents)} documents into {len(chunked_documents)} chunks.")
