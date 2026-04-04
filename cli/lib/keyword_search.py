import string
from typing import List, TypedDict
from nltk.stem import PorterStemmer
from .search_utils import get_movies, read_stopwords

stemmer = PorterStemmer()


class Movie(TypedDict):
    id: int
    title: str
    description: str


def remove_stopwords(tokens: List[str], stopwords: List[str]) -> List[str]:
    return [token for token in tokens if token not in stopwords]


def strip_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def tokenize(text: str) -> List[str]:
    return [token for token in text.split() if token.strip() != ""]


def stem_tokens(tokens: List[str]) -> List[str]:
    return [stemmer.stem(token) for token in tokens]


def preprocess_text(text: str) -> List[str]:
    stopwords = read_stopwords()
    stemmed_tokens = stem_tokens(tokenize(strip_punctuation(text.lower())))
    return remove_stopwords(stemmed_tokens, stopwords)


def has_matching_tokens(query_tokens: List[str], title_tokens: List[str]) -> bool:
    return any(token in " ".join(title_tokens) for token in query_tokens)


def search_movies(query: str, max_results: int = 5) -> List[Movie]:
    movies = get_movies()
    results = []
    preprocessed_query = preprocess_text(query)

    for movie in movies:
        preprocessed_title = preprocess_text(movie["title"])

        if has_matching_tokens(preprocessed_query, preprocessed_title):
            results.append(movie)

            if len(results) >= max_results:
                break

    return results


def print_movies(movies: List[Movie]):
    for idx, movie in enumerate(movies):
        print(f"{idx + 1}. {movie['title']}")
