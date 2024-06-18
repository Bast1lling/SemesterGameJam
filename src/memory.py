import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.memory = []

    def insert(self, user_input, llm_output):
        embedding = self.model.encode(user_input)
        container = {
            "question": user_input,
            "answer": llm_output,
            "embedding": embedding,
        }
        self.memory.append(container)

    def retrieve(self, user_input):
        # Generate the embedding for the input text
        input_embedding = self.model.encode(user_input)

        # Calculate cosine similarities
        similarities = []
        for data in self.memory:
            similarity = cosine_similarity(
                [input_embedding], [data['embedding']]
            )[0][0]
            similarities.append(similarity)

        # Find the most similar pair
        most_similar_idx = np.argmax(similarities)
        return similarities[most_similar_idx], self.memory[most_similar_idx]["answer"]


store = VectorStore()
example_data = [
    ("Hey, where am I?", "You are in a large, sumptuous dining room with silver cutlery on the table and a chandelier hanging from the ceiling."),
    ("And who am I?", "You are Jan ver Bindung, a famous detective investigating the mysterious death of a Dutch police officer with very bad breath."),
    ("I don\'t wanna die", "Sorry, I do not understand what you want from me."),
    ("But where can I go?", "You notice a grand, ornate door at the opposite end of the dining room. It seems to lead to another room."),
    ("Thank you so much!", "Sorry, I do not understand what you want from me."),
    ("Alright I go into the other room", "The living room exudes luxury and comfort. You are awed by its magnificence. Describe it")
]

for a, b in example_data:
    store.insert(a, b)

similarity, result = store.retrieve("What is this?")
print(f"With score {similarity}, this was the best answer: {result}")
similarity, result = store.retrieve("Who am I?")
print(f"With score {similarity}, this was the best answer: {result}")
similarity, result = store.retrieve("Who are you?")
print(f"With score {similarity}, this was the best answer: {result}")
similarity, result = store.retrieve("Thanks")
print(f"With score {similarity}, this was the best answer: {result}")
similarity, result = store.retrieve("Bro")
print(f"With score {similarity}, this was the best answer: {result}")
