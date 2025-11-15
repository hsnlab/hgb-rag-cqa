import time
from time import sleep, perf_counter
import traceback
from stackapi import StackAPI
import pandas as pd
import re
import math
from datetime import datetime

# SITE = StackAPI('stackoverflow', key="YOUR_KEY_HERE")
save_file_name = f"stackowerQnA.csv_{datetime.today().strftime('%Y-%m-%d')}"
# function_names = f"neo4j_function_queries.csv"
function_names = f"merged_file.csv"
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

def save_data_and_context(creation_dates, all_questions, all_question_ids, all_answers, all_answer_ids, all_contexts_a):
    data = {
        "creation_dates": creation_dates,
        "question_ids": all_question_ids,
        "questions": all_questions,
        "answer_ids": all_answer_ids,
        "answers": all_answers,
        # "question_contexts": all_contexts_q,
        "answer_contexts": all_contexts_a,
    }
    df = pd.DataFrame(data)
    df.to_csv(f"{save_file_name}+contextRIGHT.csv", index=False)



def attach_golden_context(all_answers, functions):
    href_pattern = r'href=["\'](.*?)["\']'
    code_pattern = r'<code.*?>([\s\S]*?)</code>'
    all_contexts_a = []
    for idx, answer in enumerate(all_answers):
        ans_lsts = re.findall(code_pattern, answer)
        ans_sepa = [word for s in ans_lsts for word in s.split()]
        # joined = ' '.join(ans_sepa)
        # print(joined)
        relevant_functions_a = []
        for function in functions:
            function_split = function.split(".")
            for name in function_split:
                # print(name)
                # print(re.findall(code_pattern, name))
                if name in ans_sepa:
                    # print(f"found {name}")
                    relevant_functions_a.append(function)
                    # break
            # if function[0] in ans_sepa:
            #     print(function[0])
            #     relevant_functions_a.append(function)
        # print(f"relevant_q: {relevant_functions_a}")

        all_contexts_a.append(list(set(relevant_functions_a)))
        # print(len(relevant_functions_a))
        # print(len(set(relevant_functions_a)))
        print(f"processing row: {idx + 1}")


    return all_contexts_a


def main():
    functions = pd.read_csv(function_names)
    functions = functions["f.combinedName"].tolist()
    # start = perf_counter()
    # current_time_in_epochs = math.floor(datetime.now().timestamp())
    # test_connection()
    # creation_dates, all_questions, all_question_ids, all_answers, all_answer_ids = fetch_data_weekly(300, current_time_in_epochs)


    data = pd.read_csv("stackoverflow_qna+context+time.csv")
    print(data.iloc[[0, -1]])
    creation_dates = data["creation_dates"].tolist()
    all_questions =data["questions"].tolist()
    all_question_ids =data["question_ids"].tolist()
    all_answers = data["answers"].tolist()
    all_answer_ids = data["answer_ids"].tolist()


#     all_questions = ["""<p>The output it shows, is below.</p>
# <p><a href="https://i.sstatic.net/4vYjI.jpg" rel="nofollow noreferrer">enter image description here</a></p>
# <p>I expected the output to be:</p>
# <pre><code>`KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
# metric_params=None, n_jobs=1, n_neighbors=1, p=2,
# weights='uniform')`
# </code></pre>
# <p>As I am working through the book introduction to machine learning with Python by O'Reilly.</p>
# """]
#     all_question_ids = ["75544427"]
#     all_answers = ["""<p>Just use the <code>get_params</code> method on the fitted object.</p>
# <pre><code>from sklearn.neighbors import KNeighborsClassifier
#
# X = [[0], [1], [2], [3]]
# y = [0, 0, 1, 1]
#
# neigh = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')
# neigh.fit(X, y)
#
# neigh.get_params()
# </code></pre>
# <pre><code>
# {'algorithm': 'auto',
#  'leaf_size': 30,
#  'metric': 'minkowski',
#  'metric_params': None,
#  'n_jobs': 1,
#  'n_neighbors': 1,
#  'p': 2,
#  'weights': 'uniform'}
# </code></pre>
# """]
#     all_answer_ids = ["75544689"]
#     creation_dates = ["2023-02-23 11:48:51"]



    all_contexts_a = attach_golden_context(all_answers, functions)
    save_data_and_context(creation_dates, all_questions, all_question_ids, all_answers, all_answer_ids, all_contexts_a)
    # print(f"finished in {perf_counter() - start} seconds")





if __name__ == "__main__":
    main()


