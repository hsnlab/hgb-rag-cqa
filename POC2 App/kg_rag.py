# Data
import pandas as pd
import numpy as np
import torch

import networkx as nx
import seaborn as sns
from pyvis.network import Network

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from sklearn import metrics

import spacy


class RepositoryRAG():
    llm_model = None
    model = None
    data = None
    nlp = None

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', file_path: str = 'kg_rag_df.csv', llm_model: str = "mistralai/mistral-7b-instruct-v0.3"):
        """
        Initialize the RepositoryRAG class with a SentenceTransformer model.
        
        Args:
            model_name (str): Name of the pre-trained SentenceTransformer model.
        """
        self.model = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        self.data = pd.read_csv(file_path)

        #mistral_model_id = "mistralai/Mistral-7B-Instruct-v0.1"  # or quantized: "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model, padding_side="left")
        #import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model,
        #    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device)

        self.generation_pipeline = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            device_map="auto"
        )




    def search(self, top_n: int = 10):

        answer = input("\nIs it the sklearn repository? (yes/no): ").strip().lower()
        if answer.lower() in ['yes', 'y']:
            path = '../graph/sklearn/'
        else:
            path = '../graph/manim/'
            
        
        try:
            while True:
                question = input("\nPlease enter your question (Ctrl+C to exit): ").strip()
                if not question:
                    continue

                # Step 1: Get keyword embedding
                keyword_embedding = self._extract_keywords(question)

                # Step 2: Get similarity dicts
                cluster_sims = self._cluster_retrieval(keyword_embedding, self.data['summary'].drop_duplicates().tolist())
                class_sims = self._class_retrieval(keyword_embedding, self.data['className'].drop_duplicates().tolist())

                # Step 3: Determine max similarities
                max_cluster_sim = max(cluster_sims.values(), default=0)
                max_class_sim = max(class_sims.values(), default=0)

                # Step 4: Assign weights
                cluster_weight = 2.0 if max_cluster_sim > 0.5 else 1.0
                class_weight = 2.0 if max_class_sim > 0.5 else 1.0

                # Step 5: Compute weighted similarities
                self.data['cluster_sim'] = self.data['summary'].map(cluster_sims).fillna(0) * cluster_weight
                self.data['class_sim'] = self.data['className'].map(class_sims).fillna(0) * class_weight
                self.data['total_sim'] = self.data['cluster_sim'] + self.data['class_sim']

                # Step 6: Filter and display results
                #high_sim_df = self.data[self.data['total_sim'] > 1.1].sort_values(by='total_sim', ascending=False)
#
                #if len(high_sim_df) > 10:
                #    print("\nTop results (all matches with similarity > 1.1):\n")
                #    print(high_sim_df[['combinedName', 'total_sim']].to_string(index=False))
                #else:
                top_results = self.data.sort_values(by='total_sim', ascending=False).head(top_n)
                top_functions = top_results['combinedName'].tolist()


                #context = "\n".join(f"{i+1}. {func}" for i, func in enumerate(top_functions))

                #prompt = f"""<s>[INST] You are a helpful machine learning assistant.
                #
                #    Use the following scikit-learn functions collected to answer the question. The functions order might not be fully by relavance.
                #    If unsure, say so.
                #
                #    Question: {question}
                #
                #    Relevant functions in the repository:
                #    {context}
                #
                #    Answer: [/INST]"""

                # Generate output
                #response = self.generation_pipeline(
                #    prompt,
                #    max_new_tokens=512,
                #    do_sample=True,
                #    temperature=0.3,
                #    top_p=0.95
                #)

                #answer = response[0]['generated_text'].split("[/INST]")[-1].strip()
                print("Top results:")
                print("   ")
                for item in top_functions:
                    print("  - ", item)

                print("   ")
                print("Creating visualization...")
                context_graph = self._filter_call_graph(top_functions, path = path)

                print("Generating answer...")
                answer = self._generate_answer(question, context_graph)
                print("\nAnswer:", answer)

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting search. Goodbye!")











    def _extract_keywords(self, question: str, min_token_len: int = 2) -> list:
        """
        Extract important words from a programming-related question.
        Filters out stopwords, punctuation, and selects key POS tags (nouns, verbs, adjectives).
        
        Args:
            question (str): Input question string.
            min_token_len (int): Minimum length of a word to be considered.

        Returns:
            List of important keywords.
        """
        doc = self.nlp(question)
        
        # Keep tokens that are nouns, proper nouns, verbs, adjectives
        keywords = [
            token.text for token in doc
            if token.is_alpha and not token.is_stop and len(token.text) >= min_token_len
            and token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}
        ]

        # concat keywords to one string with spaces
        keyword_string = ' '.join(keywords)

        keyword_embedding = self.model.encode(keyword_string, convert_to_tensor=True).cpu().tolist()
        
        return keyword_embedding


    def _cluster_retrieval(self, keyword_embedding, cluster_descriptions):
        """
        Retrieve the top N clusters based on cosine similarity to keyword embeddings.
        
        Args:
            keyword_embedding (list): Embeddings of the keywords.
            docstring_embeddings (list): Embeddings of the document strings.
            top_n (int): Number of top clusters to retrieve.

        Returns:
            DataFrame containing the top N clusters.
        """
        embeddings = self.model.encode(cluster_descriptions, convert_to_tensor=True).cpu().tolist()
        cosine_sim = metrics.pairwise.cosine_similarity([keyword_embedding], embeddings)[0]
        return dict(zip(cluster_descriptions, cosine_sim))


    def _class_retrieval(self, keyword_embedding, class_names):
        """
        Retrieve the top N classes based on cosine similarity to keyword embeddings.
        
        Args:
            keyword_embedding (list): Embeddings of the keywords.
            class_names (list): Embeddings of the class descriptions.
            top_n (int): Number of top classes to retrieve.

        Returns:
            DataFrame containing the top N classes.
        """
        
        embeddings = self.model.encode(class_names, convert_to_tensor=True).cpu().tolist()
        cosine_sim = metrics.pairwise.cosine_similarity([keyword_embedding], embeddings)[0]
        return dict(zip(class_names, cosine_sim))




    def _filter_call_graph(self, top_functions, path: str = '../graph/sklearn/'):
        """
        Filter the call graph based on the top functions retrieved.
        
        Args:
            top_functions (list): List of top function names to filter the call graph.
        """
        # Assuming struct_g is a DataFrame with a 'className' column
        cgn = pd.read_csv(path + 'cg_nodes.csv')
        cge = pd.read_csv(path + 'cg_edges.csv')
        sgn = pd.read_csv(path + 'sg_nodes.csv')
        sge = pd.read_csv(path + 'sg_edges.csv')
        h1e = pd.read_csv(path + 'hier_1.csv')
        h2e = pd.read_csv(path + 'hier_2.csv')


        # Filter nodes based on top functions
        cgn = cgn[cgn['combinedName'].isin(top_functions)].reset_index(drop=True)
        cge = cge[cge['source_id'].isin(cgn['func_id'].tolist()) & cge['target_id'].isin(cgn['func_id'].tolist())].reset_index(drop=True)

        sgn = sgn[sgn['func_id'].isin(cgn['func_id'].tolist())].reset_index(drop=True)
        sge = sge[sge['source_id'].isin(sgn['node_id'].tolist()) & sge['target_id'].isin(sgn['node_id'].tolist())].reset_index(drop=True)

        h1e = h1e[h1e['source_id'].isin(sgn['node_id'].tolist()) & h1e['target_id'].isin(cgn['func_id'].tolist())].reset_index(drop=True)
        h2e = h2e[h2e['source_id'].isin(cgn['func_id'].tolist()) & h2e['target_id'].isin(sgn['node_id'].tolist())].reset_index(drop=True)

        G = nx.Graph()

        # Add call graph nodes (functions)
        for _, row in cgn.iterrows():
            node_id = f"F_{row['func_id']}"
            label = f"[F] {row['combinedName']}"
            G.add_node(node_id, label=label, title=label, color="#1f78b4")  # blue

        # Add structure graph nodes
        for _, row in sgn.iterrows():
            node_id = f"S_{row['node_id']}"
            label = f"[S] {row['code'][:25]}..." if len(row['code']) > 25 else f"[S] {row['code']}"
            G.add_node(node_id, label=label, title=label, color="#33a02c")  # green

        # Add call edges
        for _, row in cge.iterrows():
            source = f"F_{row['source_id']}"
            target = f"F_{row['target_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#a6cee3")  # light blue

        # Add structure edges
        for _, row in sge.iterrows():
            source = f"S_{row['source_id']}"
            target = f"S_{row['target_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#b2df8a")  # light green

        # Add hierarchy edges (function -> structure)
        for _, row in h1e.iterrows():
            source = f"S_{row['source_id']}"
            target = f"F_{row['target_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#ff7f00", dashes=True)

        for _, row in h2e.iterrows():
            source = f"F_{row['source_id']}"
            target = f"S_{row['target_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#ff7f00", dashes=True)

        # Visualize with pyvis
        net = Network(height='1000px', width='100%', notebook=False, directed=False)
        net.from_nx(G)
        net.force_atlas_2based()
        net.save_graph('./filtered_graph.html')
        print(f"Filtered sklearn graph. visualization saved to ./filtered_graph.html")
        return G

    def _generate_answer(self, question: str, subgraph: nx.Graph):
        context_nodes = [f"{i+1}. {data.get('label', node)}"
                     for i, (node, data) in enumerate(subgraph.nodes(data=True))]
        context = "\n".join(context_nodes)

        prompt = f"""<s>[INST] You are a helpful machine learning assistant.

Use the provided context to answer the question regarding this software library. The functions order might not be fully by relavance.
If unsure, say so.

Question: {question}

Relevant functions in the repository:
{context}

Answer: [/INST]"""
        # Generate output
        response = self.generation_pipeline(
            prompt,
            max_new_tokens=512,
            return_full_text=False,
#            do_sample=True,
#            temperature=0.3,
#            top_p=0.95
        )
        answer = response[0]['generated_text'].strip()

        return answer

if __name__ == "__main__":

    tool = RepositoryRAG()
    tool.search()