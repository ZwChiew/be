from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from models.core import clean_text


class QuestionMatcher:
    print("initializing model")

    def __init__(self):
        self.model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2')
        self.annoy_index = None
        self.question_sets = []
        self.answers = []
        self.question_embeddings = []

    def initialize_annoy_index(self, question_sets):
        print("buidling ann model")
        cleaned_sentences = []
        for row in question_sets:
            cleaned_row = []  # Initialize an empty list for cleaned row
            for col in row:
                # Apply your clean_text function to each sentence
                cleaned_col = clean_text(col)
                # Append the cleaned sentence to the row
                cleaned_row.append(cleaned_col)
            # Append the cleaned row to the result
            cleaned_sentences.append(cleaned_row)
        question_sets = cleaned_sentences

        self.question_embeddings = [
            [self.model.encode(question) for question in faq] for faq in question_sets
        ]

        self.annoy_index = AnnoyIndex(
            len(self.model.encode(question_sets[0][0])), metric='euclidean')
        for idx, question_set in enumerate(question_sets, start=1):
            set_embedding = np.mean([self.model.encode(question)
                                     for question in question_set], axis=0)
            self.annoy_index.add_item(idx, set_embedding)
        self.annoy_index.build(n_trees=10)

    def update_question_sets(self, updated_question_sets):
        print("updating knowledge base for ANN")
        self.question_sets = updated_question_sets
        if self.annoy_index:
            self.initialize_annoy_index(updated_question_sets)

    def cos_sim_ind(self, idx, input):
        sample = self.question_embeddings[idx - 1]
        single_faq = []
        for s in sample:
            single_faq.append(cosine_similarity([input], [s])[0][0])
            # print(single_faq)
        return (np.mean(single_faq))

    def find_best_match(self, user_input, n_neighbors=5):
        if not self.annoy_index:
            raise ValueError(
                "Annoy Index is not initialized. Call initialize_annoy_index method first.")

        user_input_embedding = self.model.encode(clean_text(user_input))
        nearest_set_indices = self.annoy_index.get_nns_by_vector(
            user_input_embedding, n_neighbors)

        similarity_scores = []
        for idx in nearest_set_indices:
            candidate_set_embedding = self.annoy_index.get_item_vector(idx)
            # similarity = cosine_similarity([user_input_embedding], [
            #                               candidate_set_embedding])[0][0]
            similarity = self.cos_sim_ind(idx, user_input_embedding)
            similarity_scores.append((idx, similarity))

        sorted_candidates = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True)
        return sorted_candidates[0]
