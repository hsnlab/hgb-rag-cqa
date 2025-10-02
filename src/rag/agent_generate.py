from crewai import Agent, Task, Crew
from crewai.llm import BaseLLM
from typing import Optional, List
import traceback


class HFLocalLLM(BaseLLM):
    """
    Custom wrapper to use a Hugging Face transformers pipeline inside CrewAI.
    """

    def __init__(self, pipeline, max_new_tokens: int = 512, max_time: int = 120):
        super().__init__(model="hf-local")  # Dummy model name
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.max_time = max_time

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                max_time=self.max_time,
            )

            # --- DEBUG: print raw pipeline output ---
            print("\n[HFLocalLLM DEBUG] Raw pipeline output:")
            import pprint
            pprint.pprint(outputs)

            raw_text = outputs[0]["generated_text"]

            # Ha listát kaptunk (pl. role/content dict-ekkel)
            if isinstance(raw_text, list):
                assistant_parts = [
                    item["content"] for item in raw_text
                    if isinstance(item, dict) and item.get("role") == "assistant"
                ]
                text = " ".join(assistant_parts)
            else:
                text = str(raw_text)

            # Prompt kiszűrése
            if isinstance(text, str) and prompt in text:
                text = text.replace(prompt, "")

            return text.strip()

        except Exception as e:
            return f"[HFLocalLLM error: {e}]"
        
     # 👇 Required by BaseLLM, CrewAI calls this one
    def call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return self._call(prompt, stop)


def run_agents(question: str, context: dict, huggingface_apikey: Optional[str] = None) -> str:
    """
    Creates agents dynamically, runs them with Crew, and returns the final answer.
    """

    gen_pipeline = context["gen"]
    llm = HFLocalLLM(gen_pipeline)

    # Step 1: Interpreter Agent
    interpreter = Agent(
        role="Interpreter",
        goal="Interpret the user question and break it down into subtasks.",
        backstory="You are responsible for analyzing the question and planning the retrieval and summarization steps.",
        allow_delegation=True,
        verbose=True,
        llm=llm,
    )

    # Step 2: Retriever Agent
    retriever = Agent(
        role="Retriever",
        goal="Retrieve the most relevant code functions or documents from Neo4j and Qdrant based on the question.",
        backstory="You are an expert in searching code and documentation using vector search and graph queries.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # Step 3: Summarizer Agent
    summarizer = Agent(
        role="Summarizer",
        goal="Summarize the retrieved documents into a clear and concise answer to the original question.",
        backstory="You combine the retrieved information and write the final response for the user.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # Define tasks
    interpret_task = Task(
        description=f"Interpret the question: '{question}'. Break it into subtasks such as retrieval and summarization.",
        expected_output="A structured plan of subtasks and required agents.",
        agent=interpreter,
    )

    retrieve_task = Task(
        description="Retrieve the most relevant code functions or docs related to the question.",
        expected_output="A list of relevant code snippets, functions, or documentation.",
        agent=retriever,
    )

    summarize_task = Task(
        description="Summarize the retrieved documents into a final answer for the original question.",
        expected_output="A clear and concise final answer to the user's question.",
        agent=summarizer,
    )

    # Create the crew
    crew = Crew(
        agents=[interpreter, retriever, summarizer],
        tasks=[interpret_task, retrieve_task, summarize_task],
        verbose=True,
    )

    # Run the crew pipeline
    try:
        result = crew.kickoff()
        return result
    except Exception as e:
        print("❌ Error during Crew execution:")
        print(traceback.format_exc())
        return f"[Crew execution error: {e}]"
