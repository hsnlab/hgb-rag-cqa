import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class CrossAttentionFusion(nn.Module):
    def __init__(self, code_embed_dim: int = 384, graph_embed_dim: int = 128, 
                 hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        """
        Cross-attention fusion module for multimodal representation learning.
        
        Args:
            code_embed_dim (int): Dimension of code embeddings
            graph_embed_dim (int): Dimension of graph embeddings  
            hidden_dim (int): Hidden dimension for attention computations
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.code_embed_dim = code_embed_dim
        self.graph_embed_dim = graph_embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Project embeddings to common hidden dimension
        self.code_proj = nn.Linear(code_embed_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_embed_dim, hidden_dim)
        
        # Cross-attention layers
        # Code attends to graph (graph as key/value, code as query)
        self.code_to_graph_attention = MultiHeadCrossAttention(
            query_dim=hidden_dim,
            key_value_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Graph attends to code (code as key/value, graph as query)
        self.graph_to_code_attention = MultiHeadCrossAttention(
            query_dim=hidden_dim,
            key_value_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Final fusion attention layer
        self.final_fusion_attention = MultiHeadSelfAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Attentive Pooling
        self.pooler = AttentivePooler(hidden_dim)
        
        # Layer normalization
        self.ln_code = nn.LayerNorm(hidden_dim)
        self.ln_graph = nn.LayerNorm(hidden_dim)
        self.ln_final = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, code_embeddings: torch.Tensor, graph_embeddings: torch.Tensor, code_attention_mask: torch.Tensor):
        """
        Forward pass of cross-attention layer
        
        Args:
            code_embeddings: [batch_size, code_seq_len, code_embed_dim]
            graph_embeddings: [batch_size, graph_embed_dim]
            code_attention_mask: [batch_size, code_seq_len]
        Returns:
            fused_representation: [batch_size, hidden_dim]
        """
        batch_size, code_seq_len, _ = code_embeddings.shape
        
        # Project to common hidden dimension
        code_proj = self.code_proj(code_embeddings)  
        graph_proj = self.graph_proj(graph_embeddings)  
        
        # Add sequence dimension to graph for attention
        graph_proj = graph_proj.unsqueeze(1)  
        
        # Cross-attention: code attends to graph
        code_attended = self.code_to_graph_attention(
            query=code_proj,
            key=graph_proj,
            value=graph_proj
        )
        
        # Cross-attention: graph attends to code
        graph_attended = self.graph_to_code_attention(
            query=graph_proj,
            key=code_proj,
            value=code_proj
        )
        
        # Residual connections and layer normalization
        code_attended = self.ln_code(code_attended + code_proj)
        graph_attended = self.ln_graph(graph_attended + graph_proj)
        
        # Concatenate attended representations
        fused_input = torch.cat([code_attended, graph_attended], dim=1) 
        
        # Final fusion attention
        fused_output = self.final_fusion_attention(fused_input)  
        
        # Attentive pooling to get final representation
        code_mask = code_attention_mask  
        graph_mask = torch.ones((batch_size, 1), dtype=code_mask.dtype, device=code_mask.device)
        fused_mask = torch.cat([code_mask, graph_mask], dim=1)
        fused_representation = self.pooler(fused_output, fused_mask)
        
        # Final layer norm and dropout
        fused_representation = self.ln_final(fused_representation)
        fused_representation = self.dropout(fused_representation)
        
        return fused_representation


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, query_dim: int, key_value_dim: int, hidden_dim: int, 
                 num_heads: int, dropout: float = 0.1):
        """
        Multi-head cross-attention mechanism.
        
        Args:
            query_dim (int): Dimension of query vectors
            key_value_dim (int): Dimension of key and value vectors
            hidden_dim (int): Hidden dimension for attention
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_value_dim, hidden_dim)
        self.value_proj = nn.Linear(key_value_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        Forward pass of multi-head cross-attention.
        
        Args:
            query (torch.Tensor): Query tensor [batch_size, seq_len_q, query_dim]
            key (torch.Tensor): Key tensor [batch_size, seq_len_kv, key_value_dim]
            value (torch.Tensor): Value tensor [batch_size, seq_len_kv, key_value_dim]
            
        Returns:
            torch.Tensor: Attended output [batch_size, seq_len_q, hidden_dim]
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_kv = key.size(1)
        
        # Project inputs
        Q = self.query_proj(query)  # [B, seq_len_q, hidden_dim]
        K = self.key_proj(key)      # [B, seq_len_kv, hidden_dim]
        V = self.value_proj(value)  # [B, seq_len_kv, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [B, num_heads, seq_len_q, seq_len_kv]
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        # [B, num_heads, seq_len_q, head_dim]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_dim
        )
        
        # Final projection
        output = self.out_proj(attended)
        
        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Multi-head self-attention mechanism.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: Attended output [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, seq_len, embed_dim * 3]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # [B, num_heads, seq_len, seq_len]
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        # [B, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        # Final projection
        output = self.out_proj(attended)
        
        return output
    
class AttentivePooler(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))  # [H]

    def forward(self, token_embeddings: torch.Tensor, mask: torch.Tensor):
        # setting correct dtype and device for mask and query
        query = self.query.to(dtype=token_embeddings.dtype, device=token_embeddings.device)
        mask = mask.to(dtype=token_embeddings.dtype, device=token_embeddings.device)

        #print(f"Token embeddings: {token_embeddings}")
        #print(f"Query: {query}")
        #print(f"Mask: {mask}")
        # Compute raw attention scores
        scores = torch.einsum('bth,h->bt', token_embeddings, query)  # [B, T]
        scores = scores.masked_fill(mask == 0, float('-inf'))  # Mask padding
        attn_weights = F.softmax(scores, dim=1)  # [B, T]
        pooled = torch.einsum('bth,bt->bh', token_embeddings, attn_weights)  # [B, H]
        return pooled