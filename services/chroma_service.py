import chromadb
from chromadb.utils import embedding_functions

from app.config import CHROMA_PATH, COLLECTION_NAME


def init_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_func = embedding_functions.DefaultEmbeddingFunction()

    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )


def get_best_deals_same_area(area: str):
    collection = init_chroma()

    results = collection.get(
        where={"area": area}
    )

    return results["metadatas"]


def get_best_deals_avg_price_per_sqm(best_deals):
    if not best_deals:
        return None

    values = [
        deal["price_per_sqm"]
        for deal in best_deals
        if "price_per_sqm" in deal
    ]

    if not values:
        return None

    return sum(values) / len(values)