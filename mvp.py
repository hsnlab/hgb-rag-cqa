import os, sys
import getpass
import pandas as pd

# Torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW

from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData

from typing import Union, List

# Validation
sys.path.append(os.path.abspath('baseline.py'))
from baseline import evaluate_docstring, remove_docstrings_from_function

# Sentence Transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Python imports
from typing import Union, Dict, Optional, List
from tqdm import tqdm

# Disable all warnings
import warnings
warnings.filterwarnings("ignore")

# Plotting
import matplotlib.pyplot as plt

# Attention mechanism imports
from attention import CrossAttentionFusion, MultiHeadCrossAttention, MultiHeadSelfAttention, AttentivePooler



# -----------------------------
# Prompt Template
# -----------------------------
PROMPT_TEMPLATE ="""
You are a helpful assistant that writes Python docstrings for functions using best practices in clear, professional English.

Given the context below, generate a complete and well-formatted docstring. Follow this structure strictly:

- A one-line summary of what the function does.
- A description of each parameter, including type hints.
- A description of the return value, including its type.

Use this format exactly:
\"\"\"
<summary>

Args:
    param1 (type): Description.
    param2 (type): Description.

Returns:
    type: Description.
\"\"\"

If a section (e.g., Args or Returns) does not apply, omit it entirely.

Context:
"""


# -----------------------------
# Text Embedder (Docstring)
# -----------------------------

class CodeAttentionEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        A class for embedding textual data (e.g., docstrings) using SentenceTransformer models.

        Args:
            model_name (str): The name of the pre-trained sentence transformer model.
            device (str): Device to use ("cuda", "cpu", or None for auto-detect).
        """
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> List[List[float]]:
        """
        Encode one or more text strings into sentence embeddings.

        Args:
            texts (str or List[str]): Text(s) to encode.
            batch_size (int): Batch size for encoding.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=len(texts) > batch_size
        )

        return embeddings

# -----------------------------
# GNN Code Embedder (Dummy GCN)
# -----------------------------
class DocstringPredictorGNN(nn.Module):
    def __init__(self, out_channels: int = 128):
        super().__init__()

        self.convs = nn.ModuleList()

        self.convs.append(HeteroConv({
            ('function', 'calls', 'function'): SAGEConv((384, 384), 256)
        }, aggr='sum'))
        
        self.convs.append(HeteroConv({
            ('expression', 'connected_to', 'expression'): SAGEConv((768, 768), 512)
        }, aggr='sum'))

        self.convs.append(HeteroConv({
            ('expression', 'connected_to', 'expression'): SAGEConv((512, 512), 256)
        }, aggr='sum'))

        self.convs.append(HeteroConv({
            ('function', 'contains', 'expression'): SAGEConv((256, 256), out_channels),
            ('expression', 'is_in', 'function'): SAGEConv((256, 256), out_channels)
        }, aggr='sum'))

        self.convs.append(HeteroConv({
            ('function', 'calls', 'function'): SAGEConv((out_channels, out_channels), out_channels)
        }, aggr='sum'))

        self.convs.append(HeteroConv({
            ('expression', 'connected_to', 'expression'): SAGEConv((out_channels, out_channels), out_channels)
        }, aggr='sum'))

        self.convs.append(HeteroConv({
            ('expression', 'connected_to', 'expression'): SAGEConv((out_channels, out_channels), out_channels)
        }, aggr='sum'))

        self.convs.append(HeteroConv({
            ('expression', 'is_in', 'function'): SAGEConv((out_channels, out_channels), out_channels)
        }, aggr='sum'))





    def forward(self, data: HeteroData):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        for _, conv in enumerate(self.convs):
            if conv.convs.keys() == [('function', 'calls', 'function')]:
                x_dict['function'] = F.leaky_relu(conv(x_dict, edge_index_dict)['function'])
            elif conv.convs.keys() == [('expression', 'connected_to', 'expression')]:
                x_dict['expression'] = F.leaky_relu(conv(x_dict, edge_index_dict)['expression'])
            elif conv.convs.keys() == [('function', 'contains', 'expression'), ('expression', 'is_in', 'function')]:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict['function'] = F.leaky_relu(x_dict['function'])
                x_dict['expression'] = F.leaky_relu(x_dict['expression'])
            elif conv.convs.keys() == [('expression', 'isin', 'function')]:
                x_dict['function'] = F.leaky_relu(conv(x_dict, edge_index_dict)['function'])

        return x_dict['function']    
# -----------------------------
# Multimodal Fusion + LLaMA Inference
# -----------------------------
class MultimodalCommentPipeline(nn.Module):
    #def __init__(self, model:str="meta-llama/Meta-Llama-3-8B", text_embedder_model:str="all-MiniLM-L6-v2", gcn_out_channels:int=128, hf_token: str="", comb_loss_type: str = "static"):
    def __init__(self, model:str="meta-llama/Meta-Llama-3-8B", text_embedder_model:str="all-MiniLM-L6-v2", 
                 gcn_out_channels:int=128, hf_token: str="", comb_loss_type: str = "static",
                 fusion_hidden_dim: int = 256, fusion_num_heads: int = 8, fusion_dropout: float = 0.1):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Code embedder (token representation)
        self.code_embedder = CodeAttentionEmbedder(model_name=text_embedder_model, device=self.device)
        
        # GNN (graph representation)
        self.gnn = DocstringPredictorGNN(out_channels=gcn_out_channels).to(self.device)
        
        # LLM (for docstring generation only)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model, 
            torch_dtype=torch.float32,
            token=hf_token
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fusion layer
        #self.fusion = nn.Linear(384 + gcn_out_channels, self.llm.config.hidden_size).to(self.device)
        # Cross-attention fusion layer (replaces simple linear fusion)
        self.fusion = CrossAttentionFusion(
            code_embed_dim=self.llm.config.hidden_size,  
            graph_embed_dim=gcn_out_channels,
            hidden_dim=fusion_hidden_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout
        ).to(self.device)
        
        # Final projection to LLM hidden size
        self.final_proj = nn.Linear(fusion_hidden_dim, self.llm.config.hidden_size).to(self.device)

        # Attentive Pooler (optional, can be used for pooling)
        self.attentive_pool = AttentivePooler(hidden_dim=self.llm.config.hidden_size).to(self.device)

        # Ensure special token is in tokenizer
        self.special_token = "<context>"
        if self.special_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.special_token])
            self.llm.resize_token_embeddings(len(self.tokenizer))

        # Learnable weight for losses
        self.comb_loss_type = comb_loss_type
        if comb_loss_type == "weighted_scale":
            self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        elif comb_loss_type == "logvar":
            self.log_var_ce = nn.Parameter(torch.tensor(0.0))
            self.log_var_cos = nn.Parameter(torch.tensor(0.0))
        elif comb_loss_type == "gradnorm":
            pass
        
        
    def calculate_loss(self, ce_loss: torch.Tensor, cos_loss: torch.Tensor, type: str = "weighted_scale", alpha: float=0.5) -> torch.Tensor:
        if type == "static":
            comb_loss = ce_loss + alpha * cos_loss
            
            return comb_loss
        if type == "weighted_scale":
            #ce_loss_norm = ce_loss / ce_loss.detach().mean()
            #cos_loss_norm = cos_loss / cos_loss.detach().mean()
            alpha = torch.sigmoid(self.alpha_raw)
            comb_loss = alpha * cos_loss + (1 - alpha) * ce_loss

            return comb_loss
        elif type == "logvar":
            loss = (
                    (1 / (2 * torch.exp(self.log_var_ce))) * ce_loss +
                    (1 / (2 * torch.exp(self.log_var_cos))) * cos_loss +
                    0.5 * (self.log_var_ce + self.log_var_cos)
                )
            return loss
        elif type == "gradnorm":
            # TODO: Implement GradNorm loss
            raise NotImplementedError("GradNorm loss is not implemented yet.")
        else:
            raise ValueError(f"Unknown loss type: {type}. Supported types: 'weighted_scale', 'logvar', 'gradnorm'.")

    def forward(
        self,
        function_bodys: List[str],
        code_graph: HeteroData,
        code_graph_batch_mask: List[int],
        docstrings: Optional[List[str]] = None,
        mode: str = "train",
        alpha: float = 0.5,
        max_new_tokens: int = 250
    ) -> Union[Dict[str, torch.Tensor], Dict[str, List[str]]]:

        batch_size = len(function_bodys)
        device = self.device
        
        # Step 1: Tokenize prompts in batch
        prompts = [PROMPT_TEMPLATE + f" {self.special_token}" for _ in range(batch_size)]
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        input_ids = tokenized.input_ids
        prompt_embeds = self.llm.get_input_embeddings()(input_ids)

        # Step 2: Locate special token positions
        placeholder_id = self.tokenizer.convert_tokens_to_ids(self.special_token)
        placeholder_mask = (input_ids == placeholder_id)  # (B, L)

        # Sanity check: Ensure all prompts have the special token
        assert placeholder_mask.any(dim=1).all(), "Special token missing in one or more prompts."
        
        
        # Step 3: Embed all function bodies at once (batched)
        #with torch.no_grad():
        #    code_embs = torch.tensor(
        #        self.code_embedder.encode(function_bodys),
        #        dtype=torch.float32,
        #        device=device
        #    )
        max_total_len=512
        code_tokenized = self.tokenizer(
            function_bodys,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_total_len - prompt_embeds.shape[1]
        ).to(device)

        # Embed the tokenized code
        code_input_ids = code_tokenized.input_ids 
        code_attention_mask = code_tokenized.attention_mask 
        code_embs = self.llm.get_input_embeddings()(code_input_ids) 

        # Step 4: Run GNN (already batched via code_graph_batch_mask)
        graph_embs = self.gnn(code_graph)[code_graph_batch_mask].to(device)

        # Step 5: Fuse embeddings
        #fused_embs = self.fusion(torch.cat([code_embs, graph_embs], dim=-1)) 
        #  Step 5: Cross-attention fusion (replaces simple concatenation)
        fused_embs = self.fusion(code_embs, graph_embs, code_attention_mask)
        
        # Step 5.5: Project to LLM hidden size
        fused_embs = self.final_proj(fused_embs) 

        # Step 6: Replace special token embeddings with fused embeddings
        for i in range(batch_size):
            placeholder_index = placeholder_mask[i].nonzero(as_tuple=True)[0].item()
            prompt_embeds[i, placeholder_index] = fused_embs[i]
        
        
        if mode == "train" and docstrings is not None:
            # Tokenize all gold docstrings
            gold_tokenized = self.tokenizer(
                docstrings,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_total_len - prompt_embeds.shape[1]
            ).to(device)

            gold_ids = gold_tokenized.input_ids
            gold_embeds = self.llm.get_input_embeddings()(gold_ids)

            input_embeds = torch.cat([prompt_embeds, gold_embeds], dim=1)

            pad_labels = torch.full((batch_size, prompt_embeds.shape[1]), -100).to(device)
            full_labels = torch.cat([pad_labels, gold_ids], dim=1)
            
            #print(f"Input embeddings: {input_embeds.shape}\t{input_embeds}")

            outputs = self.llm(
                inputs_embeds=input_embeds,
                labels=full_labels,
                return_dict=True,
                output_hidden_states=True,
            )
            
            num_nans_logits = torch.isnan(outputs.logits).sum().item()
            if num_nans_logits>0:
              print(f"NaNs in logits: {num_nans_logits}")

            
            ce_loss = outputs.loss

            #with torch.no_grad():
            #    ref_embed = gold_embeds.mean(dim=1)
            # Attention mask (1 for real tokens, 0 for padding)
            gold_attention_mask = gold_tokenized.attention_mask  # [B, T]
            ref_embed = self.attentive_pool(gold_embeds, gold_attention_mask)  # [B, H]
            
            #gen_embed = outputs.hidden_states[-1].mean(dim=1)
            gen_embed = fused_embs

<<<<<<< HEAD
            #print(f"Shape of gen_embed: {gen_embed.shape}")
            #print(f"Shape of ref_embed: {ref_embed.shape}")

=======
>>>>>>> 596f5bbe9af41ec76b93c902680a715c14ef7cb2
            num_nans_gen_embed = torch.isnan(gen_embed).sum().item()
            num_nans_ref_enbed = torch.isnan(ref_embed).sum().item()
            if num_nans_gen_embed>0:
              print(f"NaNs in gen_embed: {num_nans_gen_embed}")
            if num_nans_ref_enbed>0:
              print(f"NaNs in ref_embed: {num_nans_ref_enbed}")
            cos_loss = 1 - F.cosine_similarity(gen_embed, ref_embed, dim=1).mean()

            # Combine losses
            #comb_loss = alpha * ce_loss + cos_loss
            #print(f"Type of cos loss: {type(cos_loss)}")
            #print(f"Type of cross entropy loss: {type(ce_loss)}")
            comb_loss = self.calculate_loss(ce_loss, cos_loss, type=self.comb_loss_type, alpha=alpha)

            return {
                "loss": comb_loss,
                "ce_loss": ce_loss,
                "cos_loss": cos_loss
            }

        elif mode == "generate":
            attention = torch.ones(prompt_embeds.shape[:-1]).to(device)

            output_ids = self.llm.generate(
                inputs_embeds=prompt_embeds,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=attention
            )

            decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            return {"generated": decoded}
            

# -----------------------------
# Training function
# -----------------------------
def freeze_parameters(model: MultimodalCommentPipeline):
    for name, param in model.named_parameters():
        if 'gnn' in name or 'fusion' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def train(model: MultimodalCommentPipeline, dataset: pd.DataFrame, graph: HeteroData, num_epochs: int = 10, lr: float = 5e-5, batch_size: int = 1):

    freeze_parameters(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad,model.parameters()), lr=lr)
    tokenizer = model.tokenizer

    graph = graph.to(model.device)


    # Feltételezve, hogy a graph egy HeteroData objektum
    function_x = graph['function'].x

    # 1. Kiszűrjük a nem nullvektoros function node-okat
    valid_CG_nodes_mask = function_x.abs().sum(dim=1) != 0
    valid_node_indices = valid_CG_nodes_mask.nonzero(as_tuple=True)[0]

    # 2. Válasszuk ki ezek 35%-át maszkolásra
    num_to_mask = int(0.35 * valid_node_indices.size(0))
    perm = torch.randperm(valid_node_indices.size(0))
    masked_nodes = valid_node_indices[perm[:num_to_mask]]
    non_masked_nodes = valid_node_indices[perm[num_to_mask:]]

    # 3. A maszkolt node-ok 60%-a megy train maskba, többi validációba
    num_masked_train = int(0.6 * masked_nodes.size(0))
    masked_train_nodes = masked_nodes[:num_masked_train]
    masked_val_nodes = masked_nodes[num_masked_train:]

    # 4. Train mask: nem maszkolt + maszkolt train node-ok
    train_nodes = torch.cat([non_masked_nodes, masked_train_nodes])

    # 5. Nullázzuk a maszkolt node-ok embeddingjeit
    function_x[masked_nodes] = 0.0

    # 6. Train és validation maszkok létrehozása
    train_mask = torch.zeros(function_x.size(0), dtype=torch.bool)
    val_mask = torch.zeros(function_x.size(0), dtype=torch.bool)
    train_mask[train_nodes] = True
    val_mask[masked_val_nodes] = True

    train_dataset = dataset[train_mask.tolist()].copy()
    train_dataset['original_index'] = train_dataset.index
    train_dataset = train_dataset.reset_index(drop=True)

    val_dataset = dataset[val_mask.tolist()].copy()
    val_dataset['original_index'] = val_dataset.index
    val_dataset = val_dataset.reset_index(drop=True)

    # Mask valid CG node docstrings
    generated_docstrings = []
    comb_losses = []
    ce_losses = []
    cos_losses = []
    val_comb_losses = []
    val_ce_losses = []
    val_cos_losses = []
    for epoch in tqdm(range(num_epochs),desc="Training..."):
        model.train()
        total_comb_loss = 0.0
        total_ce_loss = 0.0
        total_cos_loss = 0.0
        valid_samples = 0

        for idx in range(0, len(train_dataset), batch_size):

            batch_df = train_dataset.iloc[idx:idx+batch_size]
    
            # Filter invalid samples
            valid_mask = (
                batch_df["docstring"].notna() &
                (batch_df["docstring"] != "") &
                (batch_df["docstring"].astype(str).str.lower() != "nan")
            )
            batch_df = batch_df[valid_mask]
            
            if batch_df.empty:
                continue

            function_bodies = batch_df["function_code"].tolist()
            function_docstrings = batch_df["docstring"].tolist()
            graph_batch_mask_prep = batch_df['original_index'].tolist()

            # Construct graph_batch_mask
            graph_batch_mask = torch.zeros(function_x.size(0), dtype=torch.bool).to(model.device)
            graph_batch_mask[graph_batch_mask_prep] = True

            # Forward pass – generate output
            output = model(
                function_bodys=function_bodies,
                code_graph=graph,
                code_graph_batch_mask=graph_batch_mask,
                docstrings=function_docstrings,
                mode="train"
            )
            #print(output)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print("Backward prop completed. Optimizer step done.")

            total_comb_loss += loss.item()
            total_ce_loss += output["ce_loss"].item()
            total_cos_loss += output["cos_loss"].item()
            valid_samples += 1

        # Avoid division by zero if all samples were skipped
        if valid_samples == 0:
            print(f"Epoch {epoch+1}: No valid samples to train on.")
            continue
        
        # Validation phase
        #print("Starting validation")
        model.eval()
        total_val_comb_loss = 0.0
        total_val_ce_loss = 0.0
        total_val_cos_loss = 0.0
        valid_val_samples = 0
        
        generated_docstrings = []
        with torch.no_grad():
            for idx in range(0, len(val_dataset), batch_size):
                batch_df = val_dataset.iloc[idx:idx+batch_size]
    
                # Filter invalid samples
                valid_mask = (
                    batch_df["docstring"].notna() &
                    (batch_df["docstring"] != "") &
                    (batch_df["docstring"].astype(str).str.lower() != "nan")
                )
                batch_df = batch_df[valid_mask]
                
                if batch_df.empty:
                    continue

                function_bodies = batch_df["function_code"].to_list()
                function_docstrings = batch_df["docstring"].to_list()
                graph_batch_mask_prep = batch_df['original_index'].to_list()

                graph_batch_mask = torch.zeros(function_x.size(0), dtype=torch.bool).to(model.device)
                graph_batch_mask[graph_batch_mask_prep] = True

                #print("Generating with logits, for loss calculation")
                output = model(
                    function_bodys=function_bodies,
                    code_graph=graph,
                    code_graph_batch_mask=graph_batch_mask,
                    docstrings=function_docstrings,
                    mode="train"
                )

                total_val_comb_loss += output["loss"].item()
                total_val_ce_loss += output["ce_loss"].item()
                total_val_cos_loss += output["cos_loss"].item()
                valid_val_samples += 1
                
        comb_loss = total_comb_loss / valid_samples if valid_samples > 0 else 0.0
        ce_loss = total_ce_loss / valid_samples if valid_samples > 0 else 0.0
        cos_loss = total_cos_loss / valid_samples if valid_samples > 0 else 0.0
        val_comb_loss = total_val_comb_loss / valid_val_samples if valid_val_samples > 0 else 0.0
        val_ce_loss = total_val_ce_loss / valid_val_samples if valid_val_samples > 0 else 0.0
        val_cos_loss = total_val_cos_loss / valid_val_samples if valid_val_samples > 0 else 0.0

        comb_losses.append(comb_loss)
        ce_losses.append(ce_loss)
        cos_losses.append(cos_loss)
        val_comb_losses.append(val_comb_loss)
        val_ce_losses.append(val_ce_loss)
        val_cos_losses.append(val_cos_loss)
        # Print epoch results
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {comb_loss:.4f} | "
            f"CE Loss: {ce_loss:.4f} | "
            f"Cos Loss: {cos_loss:.4f}\n"
            f"Validation Loss: {val_comb_loss:.4f} | "
            f"Validation CE Loss: {val_ce_loss:.4f} | "
            f"Validation Cos Loss: {val_cos_loss:.4f}"
        )

    # --- Final Generation After Training ---
    model.eval()
    generated_docstrings = []
    val_dataset_nonan = val_dataset[
        (val_dataset['docstring'].notna()) &  # not NaN
        (val_dataset['docstring'] != "") &    # not empty string
        (val_dataset['docstring'].astype(str).str.lower() != "nan")  # not string "NaN"
    ].reset_index(drop=True)
    with torch.no_grad():
        for idx in range(0, len(val_dataset_nonan), batch_size):
            batch_df = val_dataset_nonan.iloc[idx:idx+batch_size]

            function_bodies = batch_df["function_code"].tolist()
            graph_batch_mask_prep = batch_df['original_index'].tolist()
            graph_batch_mask = torch.zeros(function_x.size(0), dtype=torch.bool).to(model.device)
            graph_batch_mask[graph_batch_mask_prep] = True

            output = model(
                function_bodys=function_bodies,
                code_graph=graph,
                code_graph_batch_mask=graph_batch_mask,
                mode="generate"
            )
            generated_docstrings.extend(output["generated"])

    # --- Evaluate Generation ---

    
    val_dataset_nonan["docstring_gen_multimodal"] = generated_docstrings
    bleu_scores, meteor_scores = zip(*[evaluate_docstring(original, generated) for original, generated in zip(val_dataset_nonan["docstring"].tolist(), val_dataset_nonan["docstring_gen_multimodal"].tolist())])
    
    val_dataset_nonan["bleu"] = bleu_scores
    val_dataset_nonan["meteor"] = meteor_scores

    print(f"Validation BLEU: {sum(bleu_scores) / len(bleu_scores):.4f} | "
            f"Validation METEOR: {sum(meteor_scores) / len(meteor_scores):.4f}")
    #alpha = torch.sigmoid(model.alpha_raw).item()
    #print(f"Cosine loss weight: {alpha:.4f} | CE loss weight: {1 - alpha:.4f}")
    return comb_losses, ce_losses, cos_losses, val_comb_losses, val_ce_losses, val_cos_losses, val_dataset_nonan


def main():
    # setting up the environment variable for HuggingFace API key
    HUGGINGFACE_API_KEY = "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" 
    os.environ["HUGGINGFACE_TOKEN"] = HUGGINGFACE_API_KEY

    # Load dataset
    dataset = pd.read_csv("graph/manim/cg_nodes.csv")  
    # Remove docstrings from the function code
    dataset["function_code"] = dataset["function_code"].apply(remove_docstrings_from_function)
    
    graph = torch.load("graph/manim/hg.pt", weights_only=False)

    # Initialize the model
    model = MultimodalCommentPipeline(
        model="mistralai/mistral-7b-instruct-v0.3",
        comb_loss_type="weighted_scale",
        fusion_hidden_dim=256,    # Hidden dimension for fusion attention
        fusion_num_heads=8,       # Number of attention heads
        fusion_dropout=0.1        # Dropout rate
    )

    comb_losses, ce_losses, cos_losses, val_comb_losses, val_ce_losses, val_cos_losses, val_dataset_nonan = train(
        model=model, 
        dataset=dataset, 
        graph=graph, 
        num_epochs=5, 
        lr=1e-3, 
        batch_size=1
    )


    # Save the model and results
    timestamp = int(pd.Timestamp.now().timestamp())
    result_dir = f"./results/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    model_gnn_path = f"{result_dir}/model_gnn.pt"
    model_fusion_path = f"{result_dir}/model_fusion.pt"
    torch.save(model.gnn, model_gnn_path)
    torch.save(model.fusion, model_fusion_path)

    # Save validation dataset
    val_dataset_nonan.to_csv(f"{result_dir}/val_dataset_nonan.csv", index=False)

    result_df = pd.DataFrame({
        "epoch": list(range(1, len(comb_losses) + 1)),
        "comb_loss": comb_losses,
        "ce_loss": ce_losses,
        "cos_loss": cos_losses,
        "val_comb_loss": val_comb_losses,
        "val_ce_loss": val_ce_losses,
        "val_cos_loss": val_cos_losses,
    })
    result_df.to_csv(f"{result_dir}/losses.csv", index=False)

    # Plotting the losses
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(comb_losses) + 1),comb_losses, label='Training Loss', color='blue')
    plt.plot(range(1, len(val_comb_losses) + 1),val_comb_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(f"{result_dir}/loss_plot.png", bbox_inches='tight', dpi=300)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(ce_losses) + 1),ce_losses, label='Training CE Loss', color='green')
    plt.plot(range(1, len(val_ce_losses) + 1),val_ce_losses, label='Validation CE Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('CE Loss')
    plt.title('Training and Validation CE Losses')
    plt.legend()
    plt.savefig(f"{result_dir}/ce_loss_plot.png", bbox_inches='tight', dpi=300)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(cos_losses) + 1),cos_losses, label='Training Cosine Loss', color='purple')
    plt.plot(range(1, len(val_cos_losses) + 1),val_cos_losses, label='Validation Cosine Loss', color='brown')
    plt.xlabel('Epochs')
    plt.ylabel('Cosine Loss')
    plt.title('Training and Validation Cosine Losses')
    plt.legend()
    plt.savefig(f"{result_dir}/cos_loss_plot.png", bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    main()