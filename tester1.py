from models.ann2 import QuestionMatcher
from firebase import returnData
from models.RetrievalQa import preprocess
import pprint
from models.paraphrase import para
from models.core import search_website, check_for_keywords, RAG_Preprocess
import traceback
from models.lc_mem import query

question_matcher = QuestionMatcher()
question_sets, answers = returnData()  # Get initial data
question_matcher.initialize_annoy_index(question_sets)
prev_question = ""
prev_answer = ""


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


# viva
def chat_engine(input_mes):
    global prev_question
    global prev_answer
    best_match_idx, best_similarity = question_matcher.find_best_match(
        input_mes)
    pprint.pprint(f"Similarity Score: {best_similarity}")
    if best_similarity < 0.725:
        input_mes = RAG_Preprocess(input_mes)
        if preprocess(input_mes):
            print("out of context")
            result = "Sorry the context information (Strata Management Act) does not provide any information on this " \
                     "matter. Please rephrase and only ask question related to Strata Matters."
        else:
            print("using RAG with memory")
            result = query(prev_question, prev_answer, input_mes)
            if check_for_keywords(result):
                result = search_website(input_mes + " Strata Malaysia")
        prev_question, prev_answer = input_mes, result
        return result

    else:
        print("using ANN")
        best_match_set = answers[best_match_idx - 1]
        prev_question, prev_answer = input_mes, best_match_set
        if "\\n" in str(best_match_set) or "\n" in str(best_match_set):
            return best_match_set
        else:
            return para(best_match_set)

