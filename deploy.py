from models.RetrievalQa import query
from models.ann2 import QuestionMatcher
from firebase import returnData
import pprint
from models.core import search_website, check_for_keywords
import traceback

question_matcher = QuestionMatcher()
question_sets, answers = returnData()  # Get initial data
question_matcher.initialize_annoy_index(question_sets)

def update():
    global answers
    updated_question_sets, updated_answer = returnData()  # Get updated data
    question_matcher.update_question_sets(updated_question_sets)
    answers = updated_answer
    print("ANN model updated !")
    return


def main(input_mes):
    try:
        output = chat_engine(input_mes)
        return True, output
    except Exception as e:
        traceback.print_exc()  # Print detailed traceback
        # Log the exception or handle it as needed
        print(f"An error occurred: {e}")
        # Return an error message to the calling code
        return False, str(e)


# andrew version
def chat_engine(input_mes):
    best_match_idx, best_similarity = question_matcher.find_best_match(
        input_mes)
    pprint.pprint(f"Similarity Score: {best_similarity}")
    if best_similarity < 0.9:
        result = query(input_mes)
        if check_for_keywords(result):
            result = search_website(input_mes + " Strata Malaysia")
        pprint.pprint(result)
        return result

    else:
        best_match_set = answers[best_match_idx - 1]
        pprint.pprint(best_match_set)
        return best_match_set

