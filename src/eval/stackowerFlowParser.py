import time
from time import sleep, perf_counter
import traceback
from stackapi import StackAPI
import pandas as pd
import re
import math
from datetime import datetime

SITE = StackAPI('stackoverflow', key="YOUR_KEY_HERE")
save_file_name = f"stackowerQnA.csv_{datetime.today().strftime('%Y-%m-%d')}"
function_names = f"neo4j_function_queries.csv"
one_week_in_epochs: int = 604800



def test_connection() -> None:
    question_id = 79779783
    question = SITE.fetch('questions', ids=[question_id], filter="withbody")
    acc_answer_id = 79779909
    answer = SITE.fetch('answers', ids=[acc_answer_id], filter="withbody")
    print(f"QUESTION: {question}")
    print(f"ANSWER: {answer}")
    print(f"----END OF TEST QUESTION----")


def fetch_data_weekly(weeks_num: int, current_time: int):
    all_questions = []
    all_question_ids = []
    all_answers = []
    all_answer_ids = []
    creation_dates = []
    save_counter = 0
    for i in range(weeks_num):
        questions_gathered = SITE.fetch(endpoint="questions", filter="withbody", sort="votes", tagged="scikit-learn", fromdate=current_time-one_week_in_epochs, todate=current_time)
        print(f"{questions_gathered['quota_remaining']}, {len(questions_gathered['items'])}")
        for question in questions_gathered['items']:
            try:
                question_id = question['question_id']
                print(f"question_id: {question_id}")
                if "accepted_answer_id" not in question:
                    print(f"no accepted answer for {question_id}")
                    continue
                # print(f"question body: {question['body']}")
                acc_answer_id = question['accepted_answer_id']
                print(f"acc_answer_id: {acc_answer_id}")
                answer = SITE.fetch("answers", ids=[acc_answer_id], filter="withbody")
                # print(f"answer: {answer}")
                # print(f"answer body: {answer['items'][0]['body']}")
                answer_body = answer["items"][0]["body"]
                creation_dates.append(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(question["creation_date"]))  )
                all_questions.append(question["body"])
                all_question_ids.append(question["question_id"])
                all_answers.append(answer_body)
                all_answer_ids.append(acc_answer_id)
                save_counter += 1
                sleep(0.25)

            except TypeError as te:
                print(f"TypeError:{te}")
                print(traceback.format_exc())
                continue
        current_time = current_time - one_week_in_epochs
        if save_counter % 20:
            save_data(creation_dates, all_questions, all_question_ids, all_answers, all_answer_ids)
    return creation_dates, all_questions, all_question_ids, all_answers, all_answer_ids


def save_data(creation_dates, all_questions, all_question_ids, all_answers, all_answer_ids):
    data = {
        "creation_dates": creation_dates,
        "question_ids": all_question_ids,
        "questions": all_questions,
        "answer_ids": all_answer_ids,
        "answers": all_answers,
    }
    df = pd.DataFrame(data)
    df.to_csv(f"{save_file_name}.csv", index=False)

def save_data_and_context(creation_dates, all_questions, all_question_ids, all_answers, all_answer_ids, all_contexts_q, all_contexts_a):
    data = {
        "creation_dates": creation_dates,
        "question_ids": all_question_ids,
        "questions": all_questions,
        "answer_ids": all_answer_ids,
        "answers": all_answers,
        "question_contexts": all_contexts_q,
        "answer_contexts": all_contexts_a,
    }
    df = pd.DataFrame(data)
    df.to_csv(f"{save_file_name}+context.csv", index=False)



def attach_golden_context(all_questions, all_answers, functions):
    href_pattern = r'href=["\'](.*?)["\']'
    code_pattern = r'<code.*?>([\s\S]*?)</code>'
    all_contexts_q = []
    all_contexts_a = []
    for idx, (answer, question) in enumerate(zip(all_answers, all_questions)):
        relevant_functions_q = []
        relevant_functions_a = []
        for function in functions:
            function_split = function.split(".")
            for name in function_split:
                if name in re.findall(code_pattern, answer) or name in re.findall(href_pattern, answer):
                    relevant_functions_a.append(function)
                    break
                if name in re.findall(code_pattern, question) or name in re.findall(href_pattern, question):
                    relevant_functions_q.append(function)
                    break
        all_contexts_a.append(relevant_functions_a)
        all_contexts_q.append(relevant_functions_q)
        print(f"processing row: {idx + 1}")


    return all_contexts_q, all_contexts_a


def add_times():
    pass

def main():
    functions = pd.read_csv(function_names)
    functions = functions["f.combinedName"].tolist()
    start = perf_counter()
    current_time_in_epochs = math.floor(datetime.now().timestamp())
    test_connection()
    creation_dates, all_questions, all_question_ids, all_answers, all_answer_ids = fetch_data_weekly(300, current_time_in_epochs)
    all_contexts_q, all_contexts_a = attach_golden_context(all_questions, all_answers, functions)
    save_data_and_context(creation_dates, all_questions, all_question_ids, all_answers, all_answer_ids, all_contexts_q, all_contexts_a)
    print(f"finished in {perf_counter() - start} seconds")





if __name__ == "__main__":
    main()


