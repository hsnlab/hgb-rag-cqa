import pandas as pd
import numpy as np
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from transformers import AutoModelForCausalLM,AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
import sys
sys.path.append("..")
from kg_rag import RepositoryRAG


####################################
#        Retrieval Metrics         #
####################################
def calculate_precision_at_k(retrieved, relevant, k):
    """
    Precision@k = (# of relevant documents in top k) / k
    """
    k = min(k, len(retrieved))
    if k <= 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    num_relevant_at_k = sum(1 for doc in retrieved_at_k if doc in relevant_set)
    return num_relevant_at_k / k


def calculate_recall_at_k(retrieved, relevant, k):
    """
    Recall@k = (# of relevant documents in top k) / (total # of relevant documents)
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    k = min(k, len(retrieved))
    retrieved_at_k = retrieved[:k]
    num_relevant_at_k = sum(1 for doc in retrieved_at_k if doc in relevant_set)
    return num_relevant_at_k / len(relevant_set)


def calculate_f1_at_k(retrieved, relevant, k):
    """
    F1@k = 2 * (precision@k * recall@k) / (precision@k + recall@k)
    """
    precision = calculate_precision_at_k(retrieved, relevant, k)
    recall = calculate_recall_at_k(retrieved, relevant, k)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_rr(retrieved, relevant):
    """
    Reciprocal Rank (RR): 1 / rank of the first relevant document.
    retrieved: ranked list of retrieved items
    relevant: set/list of relevant items (unordered)
    """
    relevant_set = set(relevant)
    for idx, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / idx
    return 0.0

def calculate_iou(retrieved, relevant, k=None):
    """
    IoU (Intersection over Union - Jaccard similarity) between retrieved and relevant sets.
    If k is given, only consider the top-k retrieved items.
    """
    if k > len(retrieved):
        k = len(retrieved)
    retrieved = retrieved[:k]
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    if not retrieved_set and not relevant_set:
        return 1.0  # edge case: both empty
    if not retrieved_set or not relevant_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(retrieved_set | relevant_set)

####################################
#    Question Answering Metrics    #
####################################

def evaluate_answer(reference: str, candidate: str, bleu_smoothing_func = SmoothingFunction().method4) -> float:
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

def evaluate_with_bertscore(reference: str, candidate: str, scorer: BERTScorer) -> float:
    """
    Compute BERTScore (F1) efficiently using a preloaded scorer.
    """
    P, R, F1 = scorer.score([candidate], [reference])
    return float(F1[0])

def evaluate_with_ragas(question: str, answer: str, reference: str, context: list[str], llm, embeddings) -> dict:
    """
    Run Ragas metrics: faithfulness, answer relevance, semantic similarity.
    """
    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [context],
        "reference": [reference]
    })
    results = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm = llm,
        embeddings = embeddings,
    )
    scores = results.to_pandas().iloc[0].to_dict()
    return scores

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def evaluate_semantic_similarity(reference: str, candidate: str, model) -> float:
    """
    Compute semantic similarity between reference and candidate text 
    using cosine similarity on sentence embeddings.
    
    Args:
        reference (str): The ground truth answer
        candidate (str): The generated answer
        model: A sentence-transformer embedding model

    Returns:
        float: cosine similarity score in [0,1]
    """
    embeddings = model.encode([reference, candidate], convert_to_numpy=True, normalize_embeddings=True)
    return cosine_similarity(embeddings[0], embeddings[1])

class RAGEvaluator:
    def __init__(self, df: pd.DataFrame, rag_model: RepositoryRAG, k_values=[3, 5, 10], eval_llm_model_name="google-t5/t5-small", eval_embed_model_name="sentence-transformers/all-MiniLM-L6-v2", huggingface_apikey = "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"):
        
        self.df = df.copy()
        self.rag = rag_model
        self.k_values = k_values
        # Initialize evaluation answer evaluation models
        os.environ["HUGGINGFACE_TOKEN"] = huggingface_apikey
        os.environ["HF_TOKEN"] = huggingface_apikey
        os.environ["HUGGINGFACE_HUB_TOKEN"] = huggingface_apikey
        ## for bertscore
        self.bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli",lang="en", rescale_with_baseline=True)
        ## for ragas metrics
        #eval_tokenizer = AutoTokenizer.from_pretrained(eval_llm_model_name)
        #eval_model = AutoModelForSeq2SeqLM.from_pretrained(eval_llm_model_name,token=huggingface_apikey)
        #eval_llm = pipeline("text2text-generation", model=eval_model, tokenizer=eval_tokenizer, max_length=512)
        #self.eval_llm_ = HuggingFacePipeline(pipeline=eval_llm)
        self.eval_embeddings = SentenceTransformer(eval_embed_model_name)

        self._prepare_columns()

    def _prepare_columns(self):
        for k in self.k_values:
            self.df[f'precision_{k}'] = None
            self.df[f'recall_{k}'] = None
            self.df[f'f1_{k}'] = None
            self.df[f'iou_{k}'] = None
        for metric in ["mrr","bleu", "meteor", "bertscore", "faithfulness", "answer_relevancy", "semantic_similarity"]:
            self.df[metric] = None


    def _run_rag(self, question: str, top_n: int):
        print("\nRetrieving top results...")

        functions_df = self.rag.retrieve_code_functions(question, top_n=top_n)
        issues_df = self.rag.retrieve_issues(question, top_n=top_n)
        prs_df = self.rag.retrieve_prs(question, top_n=top_n)

        reranked = self.rag.rerank_functions(functions_df, issues_df, prs_df, top_n=top_n)
        # Generate answer using RAG
        subgraph = self.rag._filter_knowledge_graph(reranked, issues_df, prs_df)
        answer = self.rag._generate_answer(question, subgraph)
        return list(reranked["combinedName"].values), answer

    def evaluate(self, verbose=True):
        for idx, row in self.df.iterrows():
            question = row.get("question","")
            context = row.get("edit_functions", [])
            answer = row.get("answer", "")

            if verbose:
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print(f"Golden context: {context}")

            
            top_functions, answer_gen = self._run_rag(question, top_n=max(self.k_values))
            
            if verbose:
                print(f"Retrieved functions: {top_functions}")

            for k in self.k_values:
                precision = calculate_precision_at_k(top_functions, context, k)
                recall = calculate_recall_at_k(top_functions, context, k)
                f1 = calculate_f1_at_k(top_functions, context, k)
                iou = calculate_iou(top_functions, context, k)

                self.df.at[idx, f'precision_{k}'] = precision
                self.df.at[idx, f'recall_{k}'] = recall
                self.df.at[idx, f'f1_{k}'] = f1
                self.df.at[idx, f'iou_{k}'] = iou

                if verbose:
                    print(f"Precision@{k}: {precision}, Recall@{k}: {recall}, F1@{k}: {f1}, IoU@{k}: {iou}")
            mrr = calculate_rr(top_functions, context)
            self.df.at[idx, 'mrr'] = mrr
            if verbose:
                print(f"MRR: {mrr}")
                print(f"Answer: {answer}")
                print(f"Generated Answer: {answer_gen}")
            # Evaluate answer correctness
            bleu, meteor = evaluate_answer(answer, answer_gen)
            bertscore = evaluate_with_bertscore(answer, answer_gen, self.bert_scorer)
            #ragas_metrics = evaluate_with_ragas(question, answer_gen, answer, context, self.eval_llm_, self.eval_embeddings)
            #faithfulness_score = ragas_metrics.get("faithfulness", None)
            #answer_relevancy_score = ragas_metrics.get("answer_relevancy", None)
            semantic_similarity_score = evaluate_semantic_similarity(answer, answer_gen, self.eval_embeddings)
            
            self.df.at[idx, 'bleu'] = bleu
            self.df.at[idx, 'meteor'] = meteor
            self.df.at[idx, 'bertscore'] = bertscore
            #self.df.at[idx, 'faithfulness'] = faithfulness_score
            #self.df.at[idx, 'answer_relevancy'] = answer_relevancy_score
            self.df.at[idx, 'semantic_similarity'] = semantic_similarity_score
            if verbose:
                print(f"BLEU: {bleu}, METEOR: {meteor}")
                print(f"BERTScore: {bertscore}, Semantic Similarity: {semantic_similarity_score}")
            if verbose:
                print("-" * 40)

    def print_summary(self):
        print("Evaluation metrics:")
        mrr_mean = self.df['mrr'].mean()
        print(f"MRR (all relevant): {mrr_mean:.4f}")
        for k in self.k_values:
            precision_mean = self.df[f'precision_{k}'].mean()
            recall_mean = self.df[f'recall_{k}'].mean()
            f1_mean = self.df[f'f1_{k}'].mean()
            iou_mean = self.df[f'iou_{k}'].mean()
            print(f"Precision@{k}: {precision_mean:.4f}, Recall@{k}: {recall_mean:.4f}, F1@{k}: {f1_mean:.4f}, IoU@{k}: {iou_mean:.4f}")

    def export(self, path):
        self.df.to_csv(path, index=False)
