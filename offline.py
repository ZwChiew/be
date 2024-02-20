from models.RetrievalQa import query
from deploy import main

while True:
    a = input("Please enter your question: ")
    print(main(a))

