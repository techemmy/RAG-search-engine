#!/usr/bin/env python3

import argparse
from lib.keyword_search import print_movies
from lib.inverted_index import (
    bm25_idf_command,
    build_command,
    get_idf_command,
    search_command,
    get_tf_command,
    get_tfidf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a given document and term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency for a given term"
    )
    idf_parser.add_argument("term", type=str, help="Term to search for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score for a given document and term"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to search for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search_word = args.query
            matching_movies = search_command(search_word)
            print_movies(matching_movies)
        case "build":
            build_command()
        case "tf":
            term_frequency = get_tf_command(args.doc_id, args.term)
            print(term_frequency)
        case "idf":
            idf = get_idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = get_tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score for '{args.term}' in document {args.doc_id}: {tfidf:.2f}"
            )
        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF for '{args.term}': {bm25_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
