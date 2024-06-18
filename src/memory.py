from typing import Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
        self.memory = []

    def insert(self, user_input, llm_output):
        embedding = self.model.encode(user_input)
        container = {
            "question": user_input,
            "answer": llm_output,
            "embedding": embedding,
        }
        self.memory.append(container)

    def retrieve(
        self, compare_to: str, n=3, min_similarity=0.3
    ) -> list[Tuple[str, str]]:
        # Generate the embedding for the input text
        input_embedding = self.model.encode(compare_to)

        # Calculate cosine similarities
        similarities = []
        for data in self.memory:
            similarity = cosine_similarity([input_embedding], [data["embedding"]])[0][0]
            similarities.append(similarity)

        results = []
        while len(results) < n and len(similarities) > 0:
            most_similar_idx = np.argmax(similarities)
            value = similarities.pop(most_similar_idx)
            if value < 0 or (len(results) > 0 and value < min_similarity):
                break

            result = self.memory[most_similar_idx]
            results.append((result["question"], result["answer"]))
        # Find the most similar pair
        return results
