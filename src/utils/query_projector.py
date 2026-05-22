import os
import pickle
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase


class QueryProjector(nn.Module):
    """MLP that projects e5 query embeddings into the GNN embedding space.

    Input  : (B, in_dim)   text embeddings from e5-large-v2 (1024D by default)
    Output : (B, out_dim)  L2-normalized vectors in the GNN embedding space (256D)

    The output is L2-normalized so that downstream cosine similarity is just a
    dot product against the equally normalized function GNN embeddings.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = {
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "out_dim": out_dim,
            "dropout": dropout,
        }
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project query vectors and L2-normalize the result."""
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


def collect_training_data(
    neo4j_uri: str,
    user: str,
    password: str,
    gnn_emb_path: str,
    encoder_model: str = "intfloat/e5-large-v2",
    device: str = "cpu",
    min_text_len: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Build self-supervised training pairs (query_emb, function_gnn_emb).

    For every FUNCTION node that has both a GNN embedding and a non-trivial
    `combinedName` + `docstring`, this builds the "query side" by encoding
    that text with e5-large-v2, and pairs it with the function's GNN embedding
    as the "doc side". Returns the stacked tensors plus the function ID list
    (kept in the same order as the rows of X / Y).
    """
    with open(gnn_emb_path, "rb") as f:
        gnn_embs = pickle.load(f)

    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
    rows = []
    with driver.session() as s:
        result = s.run(
            "MATCH (n:FUNCTION) "
            "RETURN n.ID AS fid, n.combinedName AS name, n.docstring AS docstring"
        )
        for r in result:
            rows.append((int(r["fid"]), r["name"] or "", r["docstring"] or ""))
    driver.close()

    pairs = []
    for fid, name, docstring in rows:
        if fid not in gnn_embs:
            continue
        text = f"{name}: {docstring}".strip(": ").strip()
        if len(text) < min_text_len:
            continue
        pairs.append((fid, text, gnn_embs[fid]))

    if not pairs:
        raise RuntimeError("No (function, gnn_embedding) pairs found.")

    encoder = HuggingFaceEmbeddings(
        model_name=encoder_model, model_kwargs={"device": device}
    )
    texts = [t for _, t, _ in pairs]
    query_vecs = encoder.embed_documents(texts)

    X = torch.tensor(query_vecs, dtype=torch.float)
    Y = torch.tensor([gnn for _, _, gnn in pairs], dtype=torch.float)
    Y = F.normalize(Y, p=2, dim=-1)
    fids = [fid for fid, _, _ in pairs]
    return X, Y, fids


def info_nce_loss(
    z_query: torch.Tensor,
    z_doc: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """In-batch InfoNCE contrastive loss for retrieval.

    Both inputs are assumed L2-normalized so that z_query @ z_doc.T equals the
    cosine similarity matrix. The diagonal entries are positives and every
    off-diagonal entry in the same row acts as a negative.
    """
    logits = z_query @ z_doc.t() / temperature
    labels = torch.arange(z_query.size(0), device=z_query.device)
    return F.cross_entropy(logits, labels)


def train_projector(
    X: torch.Tensor,
    Y: torch.Tensor,
    out_dir: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    val_frac: float = 0.1,
    temperature: float = 0.07,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[QueryProjector, float]:
    """Train QueryProjector with in-batch InfoNCE on (X, Y) pairs.

    Splits the data into train/val with `val_frac`, trains for `epochs`
    epochs, evaluates top-1 retrieval accuracy on the validation split each
    epoch, and saves the best checkpoint to `<out_dir>/projector.pt`.
    Returns the trained projector and the best validation accuracy.
    """
    os.makedirs(out_dir, exist_ok=True)

    n = X.size(0)
    perm = torch.randperm(n)
    n_val = max(1, int(n * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    X_train = X[train_idx].to(device)
    Y_train = Y[train_idx].to(device)
    X_val = X[val_idx].to(device)
    Y_val = Y[val_idx].to(device)

    projector = QueryProjector(in_dim=X.size(-1), out_dim=Y.size(-1)).to(device)
    optimizer = torch.optim.Adam(
        projector.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_acc = 0.0
    ckpt_path = os.path.join(out_dir, "projector.pt")
    history_losses: list[float] = []
    history_val_top1: list[float] = []

    for epoch in range(1, epochs + 1):
        projector.train()
        perm = torch.randperm(X_train.size(0))
        total_loss, total_n = 0.0, 0
        for i in range(0, X_train.size(0), batch_size):
            batch_idx = perm[i : i + batch_size]
            if batch_idx.numel() < 2:
                continue
            x = X_train[batch_idx]
            y = Y_train[batch_idx]
            optimizer.zero_grad()
            z = projector(x)
            loss = info_nce_loss(z, y, temperature=temperature)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
            total_n += x.size(0)
        avg_loss = total_loss / max(1, total_n)

        projector.eval()
        with torch.no_grad():
            z_val = projector(X_val)
            sim = z_val @ Y_val.t()
            preds = sim.argmax(dim=-1)
            acc = (
                (preds == torch.arange(z_val.size(0), device=device))
                .float()
                .mean()
                .item()
            )

        marker = ""
        if acc > best_val_acc:
            best_val_acc = acc
            marker = " *"
            torch.save(
                {
                    "model_state": projector.state_dict(),
                    "config": projector.config,
                },
                ckpt_path,
            )

        history_losses.append(avg_loss)
        history_val_top1.append(acc)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | val_top1={acc:.4f}{marker}")

    history = {"losses": history_losses, "val_top1": history_val_top1, "best_val": best_val_acc}
    return projector, best_val_acc, history


def load_projector(checkpoint_path: str, device: str = "cpu") -> QueryProjector:
    """Restore a trained QueryProjector from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    p = QueryProjector(
        in_dim=cfg["in_dim"],
        hidden_dim=cfg["hidden_dim"],
        out_dim=cfg["out_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    p.load_state_dict(ckpt["model_state"])
    p.eval()
    return p


@torch.no_grad()
def project_query(
    projector: QueryProjector,
    query_text: str,
    encoder: HuggingFaceEmbeddings,
    device: str = "cpu",
) -> torch.Tensor:
    """Encode a single query string with e5 and project it into the GNN space."""
    vec = encoder.embed_query(query_text)
    x = torch.tensor(vec, dtype=torch.float, device=device).unsqueeze(0)
    return projector(x).squeeze(0)
