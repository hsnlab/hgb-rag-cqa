import pandas as pd
from crewai import Agent, Task, Crew, Process, LLM
import time
import datetime


start = time.time()

#Filter out documentation related tickets
df = pd.read_csv("eval_df.csv")
initial_rows = len(df)
if 'labels' in df.columns:
    df = df[~df['labels'].astype(str).str.contains("documentation", case=False, na=False)]
filtered_rows = len(df)
print(f"Filtered out {initial_rows - filtered_rows} documentation-related rows.")

output_file = "filtered_data.csv"
df.to_csv(output_file, index=False)


# llm = LLM(
#    model="ollama/llama3",
#    #base_url="http://localhost:11434",
#    request_timeout=120
# )
#
#
# data_wrangler_agent = Agent(
#    role="Data Wrangler",
#    goal="Load data from a CSV, filter out rows related to documentation, and save the result.",
#    backstory="You are an efficient data specialist. Your primary job is to clean up datasets for other agents to process.",
#    llm=llm,
#    tools=[CsvProcessingTool()],
#    verbose=False,
# )
#
# data_prep_task = Task(
#    description="Load the 'eval_df.csv' file, filter it to remove documentation tickets, and save the clean data.",
#    expected_output="A confirmation message indicating the name of the file where the cleaned data is saved.",
#    agent=data_wrangler_agent,
# )
#
# data_prep_crew = Crew(
#    agents=[data_wrangler_agent],
#    tasks=[data_prep_task],
#    process=Process.sequential,
#    verbose=False,
# )
#
# data_prep_result = data_prep_crew.kickoff()
# print(data_prep_result)

llm = LLM(
    model="ollama/mistral",
    request_timeout = 120
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
        "You are a master at formulating questions. You can look at any piece of text and "
        "devise clear, concise, and relevant questions that probe the most important aspects of the information presented."
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
    goal="Score the generated question-and-answer pair on a scale of 1 to 10 for its human relevance and practical value.",
    backstory=(
        "You are an expert in user experience and conversational AI. You have a keen sense of what makes a question "
        "truly useful to a human trying to understand a complex technical repository. You evaluate Q&A pairs not just for "
        "accuracy, but for their intuition, clarity, and resemblance to a real user's query."
    ),
    verbose=False,
    allow_delegation=False,
    llm=llm
)

all_summaries = []
all_questions = []
all_answers = []
all_scores = []

try:
    filtered_df = pd.read_csv("filtered_data.csv")
    print(f"\n--- Starting Q&A Generation for {len(filtered_df)} rows ---")

    for index, row in filtered_df.iterrows():
        print(f"\n--- Processing Row {index + 1} ---")
        context_text = f"Problem Statement: {row['problem_statement']}\n\nPR Comments: {row['pr_comments']}"

        task_analyze = Task(
            description=f"Analyze the following text and summarize the key points:\n\n---\n{context_text}\n---",
            expected_output='A concise summary of the main technical points.',
            agent=data_analyst,
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
            agents=[data_analyst, question_generator, answer_generator, quality_assurance_agent],
            tasks=[task_analyze, task_generate_question, task_generate_answer, task_score_qa_pair],
            process=Process.sequential,
            verbose=False,
        )

        result = qa_crew.kickoff()

        all_summaries.append(task_analyze.output)
        all_questions.append(task_generate_question.output)
        all_answers.append(task_generate_answer.output)
        all_scores.append(task_score_qa_pair.output)

        print(f"--- Finished row {index + 1} ---")

except Exception as e:
    print(f"\nAn error occurred during Q&A generation: {e}")

data = {"summaries": all_summaries,
        "questions": all_questions,
        "answers": all_answers,
        "scores": all_scores,}
df = pd.DataFrame(data)
df.to_csv("generated_QnA.csv", sep='\t', encoding='utf-8', index=False, header=True)


end = time.time()
elapsed = end - start
print(f"The script ran for {elapsed} seconds, which is {datetime.timedelta(elapsed)}")