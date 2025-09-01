import re
import pandas as pd
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

class Summarizer:
    """
    Summarizes GitHub issues and PRs with different prompt templates.
    """
    def __init__(self, model_id: str = "facebook/bart-large-cnn"):
        pipeline_ = pipeline("summarization", model=model_id,
                             device_map="auto",
                             max_new_tokens = 50)

        self.llm = HuggingFacePipeline(pipeline=pipeline_)

        self.issue_prompt = PromptTemplate.from_template(self._issue_summary_template())
        self.pr_prompt = PromptTemplate.from_template(self._pr_summary_template())

        self.issue_chain = self.issue_prompt | self.llm
        self.pr_chain = self.pr_prompt | self.llm

    def _issue_summary_template(self):
        return """Summarize the following GitHub issue into 2-3 sentences.
Focus on the bug/feature request, expected vs actual behavior, and core technical details.
Ignore system information, dependency lists, and long code snippets.

### Issue Title:
{title}

### Issue Body:
{body}

### Output:
<Concise summary here>"""

    def _pr_summary_template(self):
        return """Summarize the following GitHub pull request into 2-3 sentences.
Focus on the intent of the PR, the functions/files changed, and the fix or enhancement proposed.
Ignore boilerplate, references, and long code diffs.

### PR Title:
{title}

### PR Body:
{body}

### Output:
<Concise summary here>"""

    def summarize_issue(self, title: str, body: str) -> str:
        try:
            result = self.issue_chain.invoke({"title": title, "body": body})
            return result.strip()
        except Exception as e:
            print(f"[Summarizer Issue Error] {e}")
            return ""

    def summarize_pr(self, title: str, body: str) -> str:
        try:
            result = self.pr_chain.invoke({"title": title, "body": body})
            return result.strip()
        except Exception as e:
            print(f"[Summarizer PR Error] {e}")
            return ""


class QAPairGenerator:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct", mode:str="issue",quantize=False):
        assert mode in ["issue", "pr"], "Mode must be either 'issue' or 'pr'."
        self.mode = mode
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if quantize:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",quantization_config=bnb_config,torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

        self.llm = HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            return_full_text=False,
        ))

        if mode == "issue":
            self.prompt = PromptTemplate.from_template(self._issue_prompt_template())
        else:
            self.prompt = PromptTemplate.from_template(self._pr_only_prompt_template())

        self.singlechain = self.prompt | self.llm

        multi_prompt = PromptTemplate.from_template(self._multi_candidate_prompt_template())
        self.multichain = multi_prompt | self.llm

        self.summarizer = Summarizer()

    def _issue_prompt_template(self):
        return """You are generating Q&A pairs for evaluating a code-focused question-answering system.  
Each pair must use the given GitHub issue and pull request context.  

### Context
- Issue (user-facing problem description): 
  {issue_title}
  {issue_body}

- Pull Request (title and description, developer explanation): 
  {pr_title}
  {pr_body}

- Functions changed by the PR(s): 
  {edit_functions}

### Instructions
1. Formulate **one concise, technical question** inspired mainly by the *issue text* (title/body).  
   - The question should reflect what a developer or user might ask about the problem or the fix.  
   - The question should be no longer than 1 sentence.  
   - Always end with a "?"  

2. Write a **precise answer** that draws only on the *PR description, and changed functions*.  
   - The answer should be at most 2 sentences.  
   - Always end with a period.  

3. Ensure that the question and answer are consistent:  
   - The question should be understandable from the issue perspective.  
   - The answer should show how the PR (and changed functions) addressed it.  

4. If the context does not contain enough information to make a valid Q&A pair, output:  
   - Question: Not applicable?  
   - Answer: Not enough information.  

### Output format
Question: <Your question here?>  
Answer: <Your answer here.>
""" 

    def _pr_only_prompt_template(self):
        return """You are creating Q&A pairs to evaluate a codebase question-answering system.

Use the following context from a GitHub pull request — along with any linked issue details, if available — to generate a realistic, technical question and a specific answer.

### Context
- Functions changed by the PR: {edit_functions}
- PR Title and Description (primary source for question): {pr_problem_statement}
- Issue Title and Body (optional, use to clarify the question if available): {problem_statement}

### Instructions
- Generate a **technical question** based on the **PR title, description, and changed functions**.
- If an issue is provided, use its title/body to **inform or improve the question**, but prioritize PR content.
- Write a **precise answer** using details from **PR comments, reviews, and optionally issue comments**.
- End the question with a **"?"** and the answer with a **"."**.
- If there is not enough information to make a valid Q&A pair, say:
  Question: Not applicable?
  Answer: Not enough information.

### Output format
Question: <Your technical question?>
Answer: <Your accurate and concise answer.>
"""

    def _multi_candidate_prompt_template(self):
        return """"You are analyzing context from a software project that includes:
- A pull request description and/or linked issue(s)
- A list of changed functions in the PR

Your task is to generate three self-contained Q&A pairs that distill the essence of the PR.

Goals:
- The questions should be technically meaningful, the kind that a developer, maintainer, or user might naturally ask when reviewing the PR.
- The answers should be concise, accurate, and directly supported by the provided context (avoid speculation).
- Together, the Q&A pairs should cover different angles—for example:
  - Purpose/impact (why the change was made, what it fixes or improves)
  - Implementation detail (how a function or logic changed, notable patterns or trade-offs)
  - Practical usage or consequences (how it affects users, performance, or maintenance)

Input:
- Text context (PR and issue): {context}
- Changed functions: {edit_functions}

Output format:
Candidate 1:
Question: …
Answer: …

Candidate 2:
Question: …
Answer: …

Candidate 3:
Question: …
Answer: …
"""

    def generate(self, issue_data: pd.Series, multiple_candidates=False) -> Dict[str, str]:
        try:
            edit_functions = issue_data.get("edit_functions", [])
            print(f"[QAGen] Generating Q&A for issue: {issue_data.get('url', 'Unknown URL')}")
            if len(edit_functions) > 10:
                print("[QAGen] More than 10 edit functions, truncating to 10.")
                edit_functions = edit_functions[:10]
            issue_title = issue_data.get("issue_title", "").strip()
            issue_body = issue_data.get("issue_body", "").strip()
            pr_title = issue_data.get("pr_title", "").strip()
            pr_body = issue_data.get("pr_body", "").strip()

            issue_sum = self.summarizer.summarize_issue(issue_title, issue_body) if issue_body else "No issue body provided."
            pr_sum = self.summarizer.summarize_pr(pr_title, pr_body)

            #input_vars = {
            #    "issue_title": issue_title,
            #    "issue_body": issue_sum,
            #    "edit_functions": ", ".join(edit_functions),
            #    "comments": issue_data.get("comments", ""),
            #    "pr_title": pr_title,
            #    "pr_body": pr_sum,
            #    "pr_comments": issue_data.get("pr_comments", ""),
            #}
            input_vars = {
                "context" : pr_title + " " + pr_sum + "\n" + issue_sum + "" + issue_sum,
                "edit_functions": ", ".join(edit_functions),
            }
            if multiple_candidates:
                result = self.multichain.invoke(input_vars)
                candidates = self._parse_multi_candidate_output(result)
                if candidates:
                    best = self._score_candidates_bm25(candidates, issue_body + " " + pr_body)
                    question, answer = best.get("question"), best.get("answer")
                else:
                    question, answer = "Not applicable?", "Not enough information."
            else:
                result = self.chain.invoke(input_vars)
                question, answer = self._parse_output(result)

            return {
                "question": question,
                "answer": answer,
                "context": input_vars,
                "issue_url": issue_data.get("url", "")
            }

        except Exception as e:
            print(f"[QAGen Error] {e}")
            return {
                "question": None,
                "answer": None,
                "context": {},
                "issue_url": issue_data.get("url", "")
            }

    def generate_batch(self, issues: pd.DataFrame, batch_size: int = 4) -> List[Dict[str, str]]:
        results = []

        for i in range(0, len(issues), batch_size):
            batch_df = issues.iloc[i:i+batch_size]
            batch_inputs = []

            for _, row in batch_df.iterrows():
                batch_inputs.append({
                    "problem_statement": row.get("problem_statement", ""),
                    "edit_functions": ", ".join(row.get("edit_functions", [])),
                    #"comments": row.get("comments", ""),
                    "pr_problem_statement": row.get("pr_problem_statement", ""),
                    #"pr_comments": row.get("pr_comments", ""),
                })

            try:
                #print(batch_inputs)
                torch.cuda.empty_cache()
                outputs = self.chain.batch(batch_inputs)
                print(outputs)
                for input_dict, output_text, (_, row) in zip(batch_inputs, outputs, batch_df.iterrows()):
                    q, a = self._parse_output(output_text)
                    results.append({
                        "question": q,
                        "answer": a,
                        "context": input_dict,
                        "issue_url": row.get("url", "")
                    })

            except Exception as e:
                print(f"[Batch QAGen Error] {e}")
                # Add empty results for failed batch
                for _, row in batch_df.iterrows():
                    results.append({
                        "question": None,
                        "answer": None,
                        "context": {},
                        "issue_url": row.get("url", "")
                    })

        return results
    
    def _parse_multi_candidate_output(self, result: str) -> List[Dict[str, str]]:
        candidates = []
        current = {}
        for line in result.splitlines():
            if line.strip().startswith("Candidate"):
                if current:
                    candidates.append(current)
                    current = {}
            elif line.strip().startswith("Question:"):
                current["question"] = line.split("Question:")[-1].strip()
            elif line.strip().startswith("Answer:"):
                current["answer"] = line.split("Answer:")[-1].strip()
        if current:
            candidates.append(current)
        return candidates
    
    def _parse_output(self, output: str) -> (str, str):
        # Normalize lines
        lines = output.strip().splitlines()
        question, answer = None, None
    
        # Look for lines that start with question/answer markers
        for line in lines:
            line = line.strip()
            # Match formats like: "**Question:**", "Question:", "- Question:"
            q_match = re.match(r"[*\-]*\s*Question\s*[:\-]*\s*(.*)", line, re.IGNORECASE)
            a_match = re.match(r"[*\-]*\s*Answer\s*[:\-]*\s*(.*)", line, re.IGNORECASE)
    
            if q_match:
                question = q_match.group(1).strip()
            elif a_match:
                answer = a_match.group(1).strip()
    
        # Fallbacks
        if not question:
            question = "Not applicable?"
        elif not question.endswith("?"):
            question = question.rstrip(".") + "?"
    
        if not answer:
            answer = "Not enough information."
        elif not answer.endswith("."):
            answer = answer.rstrip("?") + "."
    
        return question, answer
    
    def _simple_tokenize(self,text: str):
        return text.lower().split()

    def _score_candidates_bm25(self,candidates: list, context_text: str) -> dict:
        """
        Score candidate questions using BM25 similarity against context (functions + PR/issue text).
        Returns the best-scoring candidate.
        """
        questions = [c["question"] for c in candidates]
        # BM25 expects a list of documents; we add the context as the last "document"
        documents = questions + [context_text]
        tokenized_docs = [self._simple_tokenize(doc) for doc in documents]

        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = self._simple_tokenize(context_text)
        scores = bm25.get_scores(tokenized_query)  # similarity of each question to context
        best_idx = np.argmax(scores[:-1])  # ignore context itself
        return candidates[best_idx]

