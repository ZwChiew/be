import firebase_admin
from firebase_admin import credentials, firestore
import pprint

cred = credentials.Certificate("dataset/credentials.json")
firebase_admin.initialize_app(cred)
def returnData():
    db = firestore.client()
    users_ref = db.collection("rules")
    documents = users_ref.get()  # This retrieves all documents

    answer = []
    dataset = []
    keywords = []

# Iterate through the documents
    for doc in documents:
        doc_data = doc.to_dict()
        keywords.append(doc_data["name"])
        answer.append(doc_data["answer"])
        dataset.append(doc_data["dataset"])
    return dataset, answer
