import pandas as pd
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatOllama
from crewai.tools import BaseTool

class CsvProcessingTool(BaseTool):
    name: str = "CSV Processing Tool"
    description: str = "A tool to load, filter, and save CSV data. It specifically removes rows related to documentation."

    def _run(self, input_file: str) -> str:
        try:
            df = pd.read_csv("eval_df.csv")
            initial_rows = len(df)
            if 'labels' in df.columns:
                df = df[~df['labels'].astype(str).str.contains("documentation", case=False, na=False)]
            filtered_rows = len(df)
            print(f"Filtered out {initial_rows - filtered_rows} documentation-related rows.")

            output_file = "filtered_data.csv"
            df.to_csv(output_file, index=False)
            return f"Data has been filtered and saved to {output_file}. It now contains {filtered_rows} rows."
        except Exception as e:
            return f"An error occurred: {e}"

#TODO: tool to save the generated qa pairs as well

#TODO: agent to score the questions and answers based on a scale from 1 to 10 as how likely that this question would be asked by a user in a real world scenario

llm = ChatOllama(
    model="gpt-oss",
    base_url="http://localhost:11434"
)


data_wrangler_agent = Agent(
    role="Data Wrangler",
    goal="Load data from a CSV, filter out rows related to documentation, and save the result.",
    backstory="You are an efficient data specialist. Your primary job is to clean up datasets for other agents to process.",
    llm=llm,
    tools=[CsvProcessingTool()],
    verbose=True,
)

data_prep_task = Task(
    description="Load the 'eval_df.csv' file, filter it to remove documentation tickets, and save the clean data.",
    expected_output="A confirmation message indicating the name of the file where the cleaned data is saved.",
    agent=data_wrangler_agent,
)

data_prep_crew = Crew(
    agents=[data_wrangler_agent],
    tasks=[data_prep_task],
    process=Process.sequential,
    verbose=True,
)

data_prep_result = data_prep_crew.kickoff()
print(data_prep_result)

data_analyst = Agent(
    role='Data Analyst',
    goal='Carefully read the provided text and identify the key information and technical details.',
    backstory=(
        "You are an expert data analyst specializing in software engineering and "
        "code issue tracking. You have a knack for dissecting bug reports and understanding "
        "the core of a technical problem from a textual description."
    ),
    verbose=True,
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
    verbose=True,
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
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

all_qa_pairs = []
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
        )

        task_generate_answer = Task(
            description="Provide a clear answer to the generated question using *only* the provided text context.",
            expected_output="A single answer corresponding to the question.",
            agent=answer_generator,
            context=[task_analyze, task_generate_question],
        )

        qa_crew = Crew(
            agents=[data_analyst, question_generator, answer_generator],
            tasks=[task_analyze, task_generate_question, task_generate_answer],
            process=Process.sequential,
            verbose=True,
        )

        result = qa_crew.kickoff()
        all_qa_pairs.append(result)
        print(f"--- Result for Row {index + 1} ---\n{result}")

except Exception as e:
    print(f"\nAn error occurred during Q&A generation: {e}")

for i, qa_pair in enumerate(all_qa_pairs):
    print(f"--- Q&A Pair {i+1} ---\n{qa_pair}\n")