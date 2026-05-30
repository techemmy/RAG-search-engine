import os

from lib.keyword_search import preprocess_text, tokenize
from lib.search_utils import DEFAULT_SEARCH_LIMIT, get_movies, CACHE_DIR
import pickle
from collections import Counter
import math


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, object] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id, text):
        self.term_frequencies[doc_id] = Counter()

        for word in preprocess_text(text):
            if word not in self.index:
                self.index[word] = set()

            self.index[word].add(doc_id)
            self.term_frequencies[doc_id][word] += 1

    def get_documents(self, term) -> list[int]:
        docs = list(self.index.get(term.lower(), set()))
        return sorted(docs)

    def build(self):
        movies = get_movies()
        for movie in movies:
            self.docmap[movie["id"]] = movie
            documentText = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], documentText)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        pickle.dump(self.index, open(self.index_path, "wb"))
        pickle.dump(self.docmap, open(self.docmap_path, "wb"))
        pickle.dump(self.term_frequencies, open(self.term_frequencies_path, "wb"))

    def load(self) -> None:
        if (
            os.path.exists(self.index_path)
            and os.path.exists(self.docmap_path)
            and os.path.exists(self.term_frequencies_path)
        ):
            self.index = pickle.load(open(self.index_path, "rb"))
            self.docmap = pickle.load(open(self.docmap_path, "rb"))
            self.term_frequencies = pickle.load(open(self.term_frequencies_path, "rb"))
        else:
            raise FileNotFoundError(
                "Index files not found. Please build the index first."
            )

    def get_term_frequency(self, doc_id, term) -> int:
        tokens = tokenize(term)

        if len(tokens) != 1:
            raise ValueError("term must be a single token")

        if (doc_id not in self.term_frequencies) or (
            term not in self.term_frequencies[doc_id]
        ):
            return 0

        return self.term_frequencies[doc_id][term]

    def get_idf(self, term: str) -> float:
        tokens = preprocess_text(term)

        if len(tokens) != 1:
            raise ValueError("term must be a single token")

        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(token))

        # print(f"Total documents: {doc_count}")
        # print(f"Documents containing '{term}': {term_doc_count}")

        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        tokens = preprocess_text(term)

        if len(tokens) != 1:
            raise ValueError("term must be a single token")

        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(token))

        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    print("Inverted index built and saved successfully.")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    idx.load()

    seen, results = set(), []

    for word in tokenize(query):
        for doc_id in idx.get_documents(word):
            if doc_id not in seen:
                seen.add(doc_id)
                results.append(idx.docmap[doc_id])

                if len(results) >= limit:
                    return results

    return results


def get_tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_term_frequency(doc_id, term)


def get_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def get_tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    tf = idx.get_term_frequency(doc_id, term)
    idf = idx.get_idf(term)

    return tf * idf


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_bm25_idf(term)
