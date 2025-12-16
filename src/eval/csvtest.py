from enum import Enum
from datetime import datetime
import pandas as pd
from crewai import Agent, Task, Crew, Process, LLM
import time

MODEL = "gpt-oss:20b"


def data_filtering():
    df = pd.read_csv("eval_prs_2023-01-01.csv")
    initial_rows = len(df)
    if 'labels' in df.columns:
        df = df[~df['labels'].astype(str).str.contains("documentation", case=False, na=False)]
    filtered_rows = len(df)
    print(f"Filtered out {initial_rows - filtered_rows} documentation-related rows.")
    output_file = "filtered_data.csv"
    df.to_csv(output_file, index=False)

    df = pd.read_csv("eval_prs_2023-01-01.csv")
    initial_rows = len(df)
    if 'labels' in df.columns:
        df = df[df['labels'].astype(str).str.contains("documentation", case=False, na=False)]
    filtered_rows = len(df)
    print(f"Filtered out {initial_rows - filtered_rows} documentation-related rows.")
    output_file = "documentations.csv"
    df.to_csv(output_file, index=False)


llm = LLM(
    # model="ollama/mistral",
    model=f"ollama/{MODEL}",
    request_timeout=720,
    num_ctx_tokens=8192,
    num_resp_tokens=4096,
)


class Level(Enum):
    JUNIOR = 'junior'
    MEDIOR = 'medior'
    SENIOR = 'senior'


def run_on_level(level: Level, docs: bool = False):
    all_summaries = []
    all_questions = []
    all_answers = []
    all_scores = []
    context_text_columns = ["problem_statement", "pr_problem_statement", "pr_comments", "edit_functions", "patches"]
    all_context_problem_statements = []
    all_context_pr_statements = []
    all_context_pr_comments = []
    all_edit_functions = []
    all_patches = []

    code_analyst = Agent(
        role='Code Analyst',
        goal='From the provided code snippet, extract only the code parts, and exclude the humann conversations.',
        backstory=(
            "You are an expert software engineer with deep knowledge of code analysis and "
            "software architecture. You excel at breaking down complex code snippets into "
            "clear, concise summaries that highlight their key functions and roles within a larger system."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    data_analyst = Agent(
        role='Data Analyst',
        goal='Carefully read the provided text and identify the key information and technical details.',
        backstory=(
            "You are an expert data analyst specializing in software engineering and "
            "code issue tracking. You have a knack for dissecting bug reports and understanding "
            "the core of a technical problem from a textual description."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    question_generator = Agent(
        role='Question Generator',
        goal='Generate insightful questions based on the text provided by the Data Analyst. '
             'The questions should be answerable from the text.',
        backstory=(
            f"You have to generate a single question from the provided text like you are a {level.value} developer, "
            "who is currently learning about this topic, and would like to gain better insight on how these things work in detail. "
            "You must only use the provided context while generating the question, the use of all other former knowledge is strictly prohibited."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm,
    )

    answer_generator = Agent(
        role='Answer Generator',
        goal='Provide accurate and concise answers to the questions, based *only* on the provided text.',
        backstory=(
            "You are a meticulous and factual AI. Your sole purpose is to answer questions based on a given context. "
            "You do not invent information and stick strictly to the text you are provided."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm,
    )

    quality_assurance_agent = Agent(
        role="Quality Assurance Analyst",
        goal="Score the generated question-and-answer pair on a scale of 1 to 10 for its human relevance and practical value. The human is a {level.value} developer level.",
        backstory=(
            "You are an expert in user experience and conversational AI. You have a keen sense of what makes a question "
            "truly useful to a human trying to understand a complex technical repository. You evaluate Q&A pairs not just for "
            "accuracy, but for their intuition, clarity, and resemblance to a real user's query."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    try:
        filtered_df = pd.read_csv("collected_prsPT5.csv")
        print(f"\n--- Starting Q&A Generation for {len(filtered_df)} rows ---")

        for index, row in filtered_df.iterrows():
            if len(row[context_text_columns[4]]) > 2500:
                print(f"\n--- Skipping Row {index + 1} due to excessive length ---")
                all_summaries.append("LEN SKIPPED")
                all_questions.append("LEN SKIPPED")
                all_answers.append("LEN SKIPPED")
                all_scores.append("LEN SKIPPED")
                all_context_problem_statements.append(row[context_text_columns[0]] if row[context_text_columns[0]] is not None else "")
                all_context_pr_statements.append(row[context_text_columns[1]] if row[context_text_columns[1]] is not None else "")
                all_context_pr_comments.append(row[context_text_columns[2]] if row[context_text_columns[2]] is not None else "")
                all_edit_functions.append(row[context_text_columns[3]] if row[context_text_columns[3]] is not None else "")
                all_patches.append(row[context_text_columns[4]] if row[context_text_columns[4]] is not None else "")
                continue
        
            if "Documentation" in row["labels"]:
                print(f"\n--- Skipping Row {index + 1} due to documentation label ---")
                all_summaries.append("DOC SKIPPED")
                all_questions.append("DOC SKIPPED")
                all_answers.append("DOC SKIPPED")
                all_scores.append("DOC SKIPPED")
                all_context_problem_statements.append(row[context_text_columns[0]] if row[context_text_columns[0]] is not None else "")
                all_context_pr_statements.append(row[context_text_columns[1]] if row[context_text_columns[1]] is not None else "")
                all_context_pr_comments.append(row[context_text_columns[2]] if row[context_text_columns[2]] is not None else "")
                all_edit_functions.append(row[context_text_columns[3]] if row[context_text_columns[3]] is not None else "")
                all_patches.append(row[context_text_columns[4]] if row[context_text_columns[4]] is not None else "")
                continue
            print(f"\n--- Processing Row {index + 1} ---")
            context_text = f"Problem Statement: {row[context_text_columns[0]]}\n\n PR Problem Statement:{row[context_text_columns[1]]}\n\n PR Comments: {row[context_text_columns[2]]}\n\n Edit Functions: {row[context_text_columns[3]]}\n\n"

            task_summarise = Task(
                description=f"Analyze the following text and throw away the human conversations, keeping only the code parts:\n\n---\n Patches: {row[context_text_columns[4]]}\n\n--",
                expected_output='A concise summary of the main technical points.',
                agent=code_analyst,
            )

            task_analyze = Task(
                description=f"Analyze the following text and summarize the key points:\n\n---\n{context_text}\n---",
                expected_output='A concise summary of the main technical points.',
                agent=data_analyst,
                context=[task_summarise],
            )

            task_generate_question = Task(
                description="Based on the analysis, generate ONE question that can be answered from the text.",
                expected_output='A single question.',
                agent=question_generator,
                context=[task_analyze],
            )

            task_generate_answer = Task(
                description="Provide a clear answer to the generated question using *only* the provided text context.",
                expected_output="A single answer corresponding to the question.",
                agent=answer_generator,
                context=[task_analyze, task_generate_question],
            )

            task_score_qa_pair = Task(
                description=(
                    "Read the following question-and-answer pair. On a scale from 1 to 10, how likely is it that this exact "
                    "question would be asked by a real human who wants to gain knowledge about the repository from an all-knowing chatbot? "
                    "A score of 1 means it is highly unlikely, robotic, or not useful. A score of 10 means it is a perfect, intuitive, "
                    "and highly valuable question a human would ask."
                ),
                expected_output="A single integer score from 1 to 10.",
                agent=quality_assurance_agent,
                context=[task_analyze, task_generate_question, task_generate_answer],
            )

            qa_crew = Crew(
                agents=[code_analyst, data_analyst, question_generator, answer_generator, quality_assurance_agent],
                tasks=[task_summarise, task_analyze, task_generate_question, task_generate_answer, task_score_qa_pair],
                process=Process.sequential,
                verbose=True,
            )

            result = qa_crew.kickoff()

            all_summaries.append(task_analyze.output)
            all_questions.append(task_generate_question.output)
            all_answers.append(task_generate_answer.output)
            all_scores.append(task_score_qa_pair.output)

            if row[context_text_columns[0]] is None:
                row[context_text_columns[0]] = ""
            if row[context_text_columns[1]] is None:
                row[context_text_columns[1]] = ""
            if row[context_text_columns[2]] is None:
                row[context_text_columns[2]] = ""
            if row[context_text_columns[3]] is None:
                row[context_text_columns[3]] = ""
            if row[context_text_columns[4]] is None:
                row[context_text_columns[4]] = ""

            all_context_problem_statements.append(row[context_text_columns[0]])
            all_context_pr_statements.append(row[context_text_columns[1]])
            all_context_pr_comments.append(row[context_text_columns[2]])
            all_edit_functions.append(row[context_text_columns[3]])
            all_patches.append(row[context_text_columns[4]])

            print(f"--- Finished row {index + 1} ---")
            data = {"statements": all_context_problem_statements,
                    "pr_statements": all_context_pr_statements,
                    "comments": all_context_pr_comments,
                    "LLM_summaries": all_summaries,
                    "LLM_questions": all_questions,
                    "LLM_answers": all_answers,
                    "LLM_scores": all_scores,
                    "edit_functions": all_edit_functions,
                    "patches": all_patches
            }
            df = pd.DataFrame(data)
            df.to_csv(
                f"generated_qna_large_{MODEL.replace(':', '')}_{level.value}_on_{datetime.today().strftime('%Y-%m-%d')}.csv",
                sep='\t', encoding='utf-8', index=False, header=True)

    except Exception as e:
        print(f"\nAn error occurred during Q&A generation: {e}")

    data = {"statements": all_context_problem_statements,
            "pr_statements": all_context_pr_statements,
            "comments": all_context_pr_comments,
            "LLM_summaries": all_summaries,
            "LLM_questions": all_questions,
            "LLM_answers": all_answers,
            "LLM_scores": all_scores,
            "edit_functions": all_edit_functions,
            "patches": all_patches,
            }
    df = pd.DataFrame(data)
    df.to_csv(
        f"generated_qna_large_{MODEL.replace(':', '')}_{level.value}_on_{datetime.today().strftime('%Y-%m-%d')}.csv",
        sep='\t', encoding='utf-8', index=False, header=True)


def so_dataparse(filename):
    return pd.read_csv(filename)


def so_datasave(savefilename, contexts, df):
    df["agentic_context"] = contexts[:len(df)]
    df.to_csv(savefilename, index=False)



def stackowerflow_context_generation(dataframe):
    contexts = []
    skipctr = 0
    try:
        prevRun = pd.read_csv("stackowerQnA_agentic_partial.csv")
        contexts = prevRun["agentic_context"].tolist()
    except FileNotFoundError:
        contexts = []

    for i in range(len(dataframe)):
        contexts.append("")
    agent = Agent(
        role='Code Extractor',
        goal='Extract function and class names from code snippets',
        backstory=(
            "You are an expert software engineer. Your only job is to "
            "read text, identify code, and extract *only* the function and "
            "class names. You are precise and provide data in the exact "
            "format requested."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    for idx, row in dataframe.iterrows():
        if skipctr != 0:
            skipctr -= 1
            print(f"\n--- Skipping Row {idx + 1} ---")
            continue

        print(f"\n--- Processing Row {idx + 1} ---")
        task = Task(
            description=(
                "Analyze the following text from a Stack Overflow answer. "
                f"Here is the text:\n\n---\n{row['answers']}\n---\n\n"
                "Identify all functions and class methods. "
                "Crucially, if a function is a method of a class, extract it in the format `ClassName.method_name`. "
                "If a function is standalone (not inside a class), extract it as `function_name`."
                "If a class is standalone, extract it as `ClassName`."
            ),
            expected_output=(
                "Respond *only* with a comma-separated list of the names. "
                "Example format: MyClass.my_method, MyClass.another_method, standalone_function. "
                "If you find no names, respond with: NONE."
            ),
            agent=agent
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()

        contexts[idx] = task.output

        if idx % 20 == 0:
                so_datasave("stackowerQnA_agentic_partial.csv", contexts, dataframe)

    return contexts

def main() -> None:
    start = time.perf_counter()
    # data_filtering()
    # run_on_level(Level.JUNIOR)
    run_on_level(Level.MEDIOR)
    run_on_level(Level.SENIOR)


    # df = so_dataparse("stackowerQnA.csv")
    # contexts = stackowerflow_context_generation(df)
    # so_datasave("stackowerQnA_agentic.csv", contexts, df)

    end = time.perf_counter()
    elapsed = end - start
    print(f"The script ran for {elapsed} seconds, which is {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")


if __name__ == "__main__":
    main()