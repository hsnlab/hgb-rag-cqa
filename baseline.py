# --------------------------------- IMPORTS ---------------------------------
# Standard library imports
import os
import ast
from tqdm import tqdm

# Data imports
import pandas as pd
from datasets import Dataset
import numpy as np

# LLM imports
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# --------------------------------- FUNCTIONS ---------------------------------
def remove_docstrings_from_function(code: str) -> str:
    """
    Remove docstrings from functions in the provided code.
    
    Args:
        code (str): The Python code as a string.
    Returns:
        str: The Python code with docstrings removed.
    """
    parsed = ast.parse(code)
    for node in ast.walk(parsed):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.unparse(parsed)

def generate_docstring(function_codes: list, llm_chain, batch_size= 10) -> str:
    """
    Generate a docstring for the given function code using an LLM chain.
    
    Args:
        function_code (str): The Python function code as a string.
        llm_chain (LLMChain): The LLM chain to use for generating the docstring.
        batch_size (int): The number of function codes to process in a single batch.
    Returns:
        str: The generated docstrings.
    """
    outputs = []
    for i in tqdm(range(0, len(function_codes), batch_size),desc="Generating docstrings..."):
        batch_end = min(i + batch_size, len(function_codes))
        batch_inputs =[
            {"function_code": code} for code in function_codes[i:batch_end]
        ]
        batch_outputs = llm_chain.batch(batch_inputs)
        #batch_outputs = [x["text"] for x in batch_outputs]
        outputs.extend(batch_outputs)
    return outputs

def generate_docstring_batch(batch, llmchain):
    prompts = [{"function_code":code} for code in batch["function_code"]]
    outputs = llmchain.batch(prompts)
    return {"mistral_7b_instruct_v3_generated_docstring":outputs}

def evaluate_docstring(reference: str, candidate: str, bleu_smoothing_func = SmoothingFunction().method4) -> float:
    """
    Calculate the BLEU- and Meteor score between a reference and a candidate docstring.
    
    Args:
        reference (str): The reference sentence.
        candidate (str): The candidate sentence.
        smoothing_func: Smoothing function to use for BLEU score calculation.
    Returns:
        bleu (float): The BLEU score.
        meteor (float): The Meteor score.
    """
    # Tokenize reference and candidate
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)
    # Calculate BLEU score
    bleu = sentence_bleu([reference_tokens], candidate_tokens,smoothing_function=bleu_smoothing_func)
    # Calculate Meteor score
    meteor = meteor_score([reference_tokens], candidate_tokens)

    return bleu, meteor

def evaluate_docstring_batch(batch):
    bleu_scores = []
    meteor_scores = []
    for ref, gen in zip(batch["docstring"], batch["mistral_7b_instruct_v3_generated_docstring"]):
        ref_tokens = word_tokenize(ref)
        gen_tokens = word_tokenize(gen)
        bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=SmoothingFunction().method4)
        meteor = meteor_score([ref_tokens], gen_tokens)
        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
    return {
        "bleu_score": bleu_scores,
        "meteor_score": meteor_scores
    }
# --------------------------------- MAIN FUNCTION ---------------------------------
def main():
    # setting up the environment variable for HuggingFace API key
    HUGGINGFACE_API_KEY = "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" 
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY


    # Load the dataset
    data_dir = "graph/sklearn/"
    dataset_path = f"{data_dir}cg_nodes.csv"
    #code_df = load_dataset("csv", data_files=dataset_path)["train"]
    code_df = pd.read_csv(dataset_path)
    
    # Remove docstrings from the function code
    #code_df = code_df.map(remove_docstrings_from_function, input_columns=["function_code"])
    code_df["function_code"] = code_df["function_code"].apply(remove_docstrings_from_function)
    code_df_hf = Dataset.from_pandas(code_df)

    # Define the prompt template
    template = """
    You are a helpful assistant that writes Python docstrings following best practices.

    Given the function code below, generate a well-formatted docstring using the following structure:
    - A one-line summary of what the function does.
    - A description of each parameter (if any).
    - A description of the return value (if any).

    Use the format:
    \"\"\"<summary>

    Args:
        param1 (type): Description.
        param2 (type): Description.

    Returns:
        type: Description.
    \"\"\"

    Function Code:
    {function_code}

    Write the docstring below:
    """

    prompt = PromptTemplate.from_template(template)

    # Download the LLM model
    model_id = "mistralai/mistral-7b-instruct-v0.3"
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
    )

    # Create the text generation pipeline
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        return_full_text=False,
    )

    # Create the LLM instance
    llm = HuggingFacePipeline(pipeline=text_generator,device_map="auto")

    # Create the LLM chain
    llm_chain = prompt | llm 
    
    # Generate docstrings for each function code
    #generated_docstrings = generate_docstring(code_df["function_code"].tolist(), llm_chain)
    code_df_hf = code_df_hf.map(generate_docstring_batch, batched=True, batch_size=10, fn_kwargs={"llm_chain": llm_chain})
    
    # Calculate BLEU scores for the generated docstrings
    # Make sure the correct nltk tokenoizers are downloaded
    import nltk
    nltk.download('wordnet')
    nltk.download('punkt_tab')

    # Filtering out rows with None or empty docstrings
    filtered_hf = code_df_hf.filter(lambda x: x["docstring"] is not None and str(x["docstring"]).strip().lower() not in ["", "nan"])
    # Score calculation
    filtered_hf = filtered_hf.map(evaluate_docstring_batch, batched=True, batch_size=10)
    filtered_pd = filtered_hf.to_pandas()

    # Add scores to the original DataFrame
    code_df = code_df_hf.to_pandas()
    code_df = pd.merge(code_df,filtered_pd[["func_id", "bleu_score", "meteor_score"]], on="func_id", how="left")

    
    # report average scores
    print(f"Average BLEU score (with {model_id}): {code_df["bleu_score"].mean()}")
    print(f"Average Meteor score (with {model_id}): {code_df["meteor_score"].mean()}")

    # Save the results to a CSV file
    output_path = f"{data_dir}cg_nodes_with_gen_docstrings.csv"
    code_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()