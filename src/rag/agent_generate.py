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
            system_prompt = (
                "You are an expert Python programming assistant. "
                "Always answer clearly and directly to the user's question. "
                "Ignore process instructions like 'Thought', 'Action', or 'Observation'. "
                "Your output should only contain the explanation or answer."
            )

            if isinstance(prompt, str):
                prompt = f"{system_prompt}\n\n{prompt}"

            # CrewAI sometimes passes a list of messages (role/content)
            if isinstance(prompt, list):
                prompt = "\n".join(
                    f"{p.get('role', '')}: {p.get('content', '')}"
                    if isinstance(p, dict) else str(p)
                    for p in prompt
                )

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

            # --- Validate output ---
            if not outputs or "generated_text" not in outputs[0]:
                print("[HFLocalLLM ERROR] Empty or invalid pipeline output.")
                return "[HFLocalLLM error: empty model response]"

            raw_text = outputs[0]["generated_text"]
            if not raw_text or not str(raw_text).strip():
                print("[HFLocalLLM ERROR] Model returned an empty string.")
                return "[HFLocalLLM error: invalid or empty text response]"

            # Extract assistant responses (if structured)
            if isinstance(raw_text, list):
                text = " ".join(
                    item["content"]
                    for item in raw_text
                    if isinstance(item, dict) and item.get("role") == "assistant"
                )
            else:
                text = str(raw_text)
            text = text.strip()

            # --- Try to extract the clean "Final Answer" block ---
            final_answer = ""
            if "Final Answer:" in text:
                final_answer = text.split("Final Answer:", 1)[-1].strip()
            elif "Final answer:" in text:
                final_answer = text.split("Final answer:", 1)[-1].strip()
            elif "Thought:" in text:
                parts = text.split("Thought:")
                final_answer = parts[-1].strip()
            else:
                final_answer = text.strip()

            # --- Cleanup step: remove artifacts and prompt leftovers ---
            clean_lines = []
            for line in final_answer.splitlines():
                lower = line.lower().strip()
                if any(skip in lower for skip in [
                    "your final answer must",
                    "i must use these formats",
                    "begin!",
                    "current task:",
                    "this is the expected criteria",
                    "you must return",
                    "thought:",
                    "```",
                ]):
                    continue
                if lower.startswith(("system:", "user:", "tool name", "tool arguments")):
                    continue
                if not line.strip():
                    continue
                clean_lines.append(line.strip())

            final_answer = "\n".join(clean_lines).strip()

            if not final_answer:
                final_answer = "[No meaningful content generated — model may have echoed prompt instructions.]"

            return final_answer

        except Exception as e:
            print("[HFLocalLLM ERROR] Exception in _call():", e)
            return f"[HFLocalLLM error: {e}]"
        
     # Required by BaseLLM, CrewAI calls this one
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
        verbose=False,
        llm=llm,
    )

    # Step 2: Retriever Agent
    retriever = Agent(
        role="Retriever",
        goal="Retrieve the most relevant code functions, documentation, or built-in explanations related to the question.",
        backstory="You are an expert at finding relevant Python documentation and examples from your internal knowledge base.",
        allow_delegation=False,
        verbose=False,
        llm=llm,
    )

    # Step 3: Summarizer Agent
    summarizer = Agent(
        role="Summarizer",
        goal="Write the final clear and concise explanation to the user's question, using the information provided.",
        backstory="You are a helpful software engineering assistant who gives accurate and complete explanations.",
        allow_delegation=False,
        verbose=False,
        llm=llm,
    )

    # Define tasks
    interpret_task = Task(
        description=f"Interpret the user's question: '{question}'. "
                "Break it into subtasks (retrieval, summarization).",
        expected_output="A structured plan of subtasks and required agents.",
        agent=interpreter,
    )

    retrieve_task = Task(
        description=f"Retrieve the most relevant Python code functions, "
                f"documentation, or explanations that help answer the question: '{question}'.",
        expected_output="Relevant Python documentation or code references.",
        agent=retriever,
    )

    summarize_task = Task(
        description=f"Based on the retrieved information, explain clearly and concisely: '{question}'.",
        expected_output="A clear and correct final explanation.",
        agent=summarizer,
    )

    # Create the crew
    crew = Crew(
        agents=[interpreter, retriever, summarizer],
        tasks=[interpret_task, retrieve_task, summarize_task],
        verbose=True,
    )

    # --- Execute ---
    try:
        result = crew.kickoff()

        # Try to access any valid output field from CrewAI
        final_output = getattr(result, "output", None) or \
                       getattr(result, "output_text", None) or \
                       getattr(result, "final_output", None)

        if not final_output:
            print("[HFLocalLLM WARNING] No valid output field found in Crew result object.")
            final_output = "[Crew execution error: missing final output]"

        # --- Try to isolate the final answer if model includes boilerplate ---
        if isinstance(final_output, str):
            lower = final_output.lower()
            if "final answer" in lower:
                idx = lower.rfind("final answer")
                final_output = final_output[idx + len("final answer"):].strip(": \n\t`")

        final_output = "\n".join([
            line for line in final_output.splitlines()
            if not any(k in line.lower() for k in ["thought:", "action:", "observation", "context"])
        ]).strip()

        print("\n=== Final Answer ===")
        print(final_output)
        return result
    
    except Exception as e:
        print("❌ Error during Crew execution:")
        print(traceback.format_exc())
        return f"[Crew execution error: {e}]"
