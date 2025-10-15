from crewai.llm import BaseLLM
from typing import Optional, List


def clean_final_output(text) -> str:
    """
    Utility function to clean CrewAI's raw output into a readable final answer.
    """
    if text is None:
        return ""

    s = str(text).strip()
    if not s:
        return s

    # Extract 'Final Answer:' block if present
    lower = s.lower()
    if "final answer:" in lower:
        idx = lower.rfind("final answer:")
        s = s[idx + len("final answer:"):].strip()

    # Remove obvious prompt/control leftovers
    junk = [
        "your job depends on it!",
        "begin!",
        "```",
        "thought:",
        "current task:",
        "this is the expected criteria",
        "you must return",
    ]
    for j in junk:
        s = s.replace(j, "")

    lines = []
    for line in s.splitlines():
        l = line.strip()
        if l.lower().startswith(("system:", "user:", "tool name", "tool arguments", "observation:")):
            continue
        if not l:
            continue
        lines.append(l)
    return "\n".join(lines).strip()


class HFLocalLLM(BaseLLM):
    """
    Simple Hugging Face pipeline wrapper compatible with CrewAI's BaseLLM.
    Used to call a local transformer model as CrewAI's reasoning backend.
    """

    def __init__(self, pipeline, max_new_tokens: int = 2048, max_time: int = 600):
        super().__init__(model="hf-local")
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.max_time = max_time

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            system_prompt = (
                "You are an expert Python programming assistant. "
                "Always answer clearly and directly to the user's question. "
                "Ignore internal reasoning markers like 'Thought', 'Action', or 'Observation'. "
                "Your output should only contain the explanation or final answer."
            )

            # Normalize prompt (can be string or list of messages)
            if isinstance(prompt, list):
                prompt = "\n".join(
                    f"{p.get('role', '')}: {p.get('content', '')}"
                    if isinstance(p, dict) else str(p)
                    for p in prompt
                )
            elif not isinstance(prompt, str):
                prompt = str(prompt)

            prompt = f"{system_prompt}\n\n{prompt.strip()}"

            # Run generation
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

            if not outputs or "generated_text" not in outputs[0]:
                return "[HFLocalLLM error: empty model response]"

            raw_text = outputs[0]["generated_text"]
            return clean_final_output(raw_text)

        except Exception as e:
            print("[HFLocalLLM ERROR]", e)
            return f"[HFLocalLLM error: {e}]"

    def call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return self._call(prompt, stop)
