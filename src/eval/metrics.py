import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import BERTScorer
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset


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
