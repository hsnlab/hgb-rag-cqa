from neo4j import GraphDatabase
from qdrant_client import models
from langchain_core.documents import Document

Q_MAP = {
    "general_question": "general",
    "bug_report": "bug report/issue/PR",
    "performance_issue": "performance",
    "feature_request": "feature request",
    "default": "general",
}

# node labels that we consider to have retrievable text in Qdrant
RETRIEVABLE_TYPES = {"FUNCTION", "ISSUE", "PR", "CLUSTER"}


def build_shortest_path_context(
    query: str,
    uri: str,
    auth: tuple[str, str],
    vectorstore,
    question_type: str,
    top_nodes: list[str],
    top_k_paths: int = 10,
    top_k_docs_per_path: int = 10,
    verbose: bool = False
) -> tuple[list[Document], list[str]]:
    """
    Build context by:
      - finding shortest paths from QUESTION(type=question_type) → each node in top_nodes,
      - creating a readable structural text,
      - retrieving relevant docs (function code, issues, PRs, etc.) for those nodes.

    Returns:
        context_docs: list[Document]  (textual context for LLM)
        path_texts:   list[str]       (structural/graph context for LLM)
    """
    if not top_nodes:
        if verbose:
            print("[INFO] No top_nodes provided, skipping graph context.")
        return [], []

    driver = GraphDatabase.driver(uri, auth=auth)
    context_docs, path_texts, seen_paths = [], [], set()

    q_type = Q_MAP.get(question_type, "general")

    with driver.session() as session:
        # --- Step 1: resolve the QUESTION node (global_id) for this query type
        q_record = session.run(
            "MATCH (q:QUESTION {type: $t}) RETURN q.global_id AS qid, q.type AS qtype",
            t=q_type,
        ).single()

        if not q_record:
            if verbose:
                print(f"[WARN] No QUESTION node found for type '{q_type}'")
            driver.close()
            return [], []

        question_global_id = q_record["qid"]
        question_display_type = q_record["qtype"]  # e.g. "performance"

        # --- Step 2: for each top node, get shortest path QUESTION → node
        for node_id in top_nodes[:top_k_paths]:
            if node_id == question_global_id:
                continue

            cypher = """
            MATCH (start {global_id: $qid})
            MATCH (end   {global_id: $nid})
            MATCH p = shortestPath((start)-[*..6]-(end))
            RETURN
                [r IN relationships(p) | {
                    rel_type: type(r),
                    src: {id: startNode(r).global_id, labels: labels(startNode(r))},
                    dst: {id: endNode(r).global_id, labels: labels(endNode(r))}
                }] AS rel_triplets,
                [n IN nodes(p) | {
                    id: n.global_id,
                    type: labels(n)[0],
                    data: {
                        name: coalesce(
                            CASE
                                WHEN 'FUNCTION' IN labels(n) THEN n.combinedName
                                WHEN 'CLASS'    IN labels(n) THEN n.name
                                WHEN 'FILE'     IN labels(n) THEN n.name
                                ELSE n.name
                            END,
                            "unknown"
                        ),
                        title: CASE
                            WHEN 'ISSUE' IN labels(n) OR 'PR' IN labels(n)
                            THEN n.title ELSE null END,
                        path: CASE
                            WHEN 'FILE' IN labels(n)
                            THEN n.path ELSE null END,
                        import_from: CASE
                            WHEN 'IMPORT' IN labels(n)
                            THEN n.import_from ELSE null END,
                        import_name: CASE
                            WHEN 'IMPORT' IN labels(n)
                            THEN n.import_name ELSE null END,
                        base_classes: CASE
                            WHEN 'CLASS' IN labels(n)
                            THEN n.base_classes ELSE null END,
                        summary: CASE
                            WHEN 'CLUSTER' IN labels(n)
                            THEN n.summary ELSE null END
                    }
                }] AS path_nodes
            """

            record = session.run(
                cypher,
                qid=question_global_id,
                nid=node_id,
            ).single()

            if not record:
                if verbose:
                    print(f"[DEBUG] No path found between QUESTION and {node_id}")
                continue

            path_nodes = record["path_nodes"]
            rel_triplets = record["rel_triplets"]
            if not path_nodes or not rel_triplets:
                continue

            # --- Step 3: build structural path string with cleanup
            path_text = _build_structural_path_text(
                path_nodes,
                rel_triplets,
                question_display_type,
            )

            # de-dupe whole-path strings across targets
            if path_text in seen_paths:
                continue
            seen_paths.add(path_text)
            path_texts.append(path_text)

            if verbose:
                print(f"[DEBUG] Added path with {len(path_nodes)} nodes.")

            # --- Step 4: retrieve docs from retrievable nodes on this path
            # collect numeric node IDs for retrievable node types only
            node_pairs = [
                (n["id"], n.get("type", "UNKNOWN"))
                for n in path_nodes
                if n.get("id") and ":" in n["id"]
            ]

            node_numeric_ids = [
                str(gid.split(":")[1])
                for gid, label in node_pairs
                if label.upper() in RETRIEVABLE_TYPES
            ]

            # avoid empty / degenerate cases
            if not node_numeric_ids:
                if verbose:
                    print(f"[DEBUG] Skipping doc fetch for path (no retrievable IDs).")
                continue

            if verbose:
                print(f"[DEBUG] numeric_ids used for Qdrant: {node_numeric_ids}")

            q_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.node_id",
                        match=models.MatchAny(any=node_numeric_ids),
                    ),
                    models.FieldCondition(
                        key="metadata.type",
                        match=models.MatchAny(
                            any=[
                                "function_code",
                                "function_docstring",
                                "issue_body",
                                "issue_title",
                                "pr_body",
                                "pr_title",
                                "semantic_cluster",
                            ]
                        ),
                    ),
                ]
            )

            path_docs = vectorstore.search(
                query=query,
                top_k=top_k_docs_per_path,
                filter=q_filter,
            )

            if verbose:
                print(f"[DEBUG] Added path documents with {len(path_docs)} docs.")

            context_docs.extend(path_docs)

    driver.close()
    return context_docs, path_texts


def _build_structural_path_text(
    path_nodes: list[dict],
    rel_triplets: list[dict],
    question_display_type: str,
) -> str:
    """
    Turn a path into a readable string like:
    (QUESTION:0 {type=performance}) -[QUESTION_CLUSTER]->
    (CLUSTER:4 {...}) -[CLUSTER_FUNCTION]->
    (FUNCTION:2288 {name=test_kernel_pca_pandas_output})

    Also:
      - collapse consecutive duplicate nodes
      - prettify QUESTION/ISSUE/PR labels
    """

    # Index nodes by global_id for lookup
    node_by_id = {n["id"]: n for n in path_nodes}

    # Build ordered edge steps: src -> rel_type -> dst
    steps = []
    for rel in rel_triplets:
        src_id = rel["src"]["id"]
        dst_id = rel["dst"]["id"]
        rel_type = rel["rel_type"]

        src_node = node_by_id.get(src_id)
        dst_node = node_by_id.get(dst_id)
        if not src_node or not dst_node:
            continue

        steps.append((src_node, rel_type, dst_node))

    rendered_parts = []
    last_node_global_id = None

    for (src_node, rel_type, dst_node) in steps:
        # Render src node if it's new
        if src_node["id"] != last_node_global_id:
            rendered_parts.append(_format_node_for_path(src_node, question_display_type))
            last_node_global_id = src_node["id"]

        # Render edge
        rendered_parts.append(f"-[{rel_type}]->")

        # Render dst node
        if dst_node["id"] != last_node_global_id:
            rendered_parts.append(_format_node_for_path(dst_node, question_display_type))
            last_node_global_id = dst_node["id"]
        else:
            # dst same as src after dedupe: skip node text
            pass

    # Join with spaces for readability
    return " ".join(rendered_parts)


def _format_node_for_path(node: dict, question_display_type: str) -> str:
    """
    Pretty-print a node for structural context.
    - QUESTION nodes: show type (e.g. performance / general / feature request)
    - ISSUE/PR nodes: prefer title
    - FUNCTION nodes: prefer name (combinedName)
    - CLUSTER nodes: show summary (trimmed)
    - Avoid ugly duplicates like QUESTION:QUESTION:0 -> just QUESTION:0
    """
    raw_type = node.get("type", "UNKNOWN")           # e.g. "QUESTION", "CLUSTER"
    raw_gid = node.get("id", "unknown")              # e.g. "QUESTION:0"
    data = node.get("data", {}) or {}

    # Clean display_id: don't repeat the label twice
    # e.g. if type="QUESTION" and id="QUESTION:0", display_id -> "QUESTION:0"
    display_id = raw_gid

    # Build human-facing payload
    # QUESTION -> use query type
    if raw_type.upper() == "QUESTION":
        display_fields = {
            "type": question_display_type
        }
    # ISSUE / PR -> title
    elif raw_type.upper() in ("ISSUE", "PR"):
        display_fields = {
            "title": data.get("title") or data.get("name") or "unknown"
        }
    # FUNCTION -> name
    elif raw_type.upper() == "FUNCTION":
        display_fields = {
            "name": data.get("name") or "unknown"
        }
    # CLUSTER -> summary (trim long), optional name
    elif raw_type.upper() == "CLUSTER":
        summary = data.get("summary") or ""
        if isinstance(summary, str) and len(summary) > 140:
            summary = summary[:300].rstrip() + "..."
        display_fields = {
            "summary": summary or "cluster",
        }
    else:
        # fallback: include whatever looks informative
        display_fields = {}
        if data.get("name"):
            display_fields["name"] = data["name"]
        if data.get("title"):
            display_fields["title"] = data["title"]
        if data.get("path"):
            display_fields["path"] = data["path"]

        if not display_fields:
            display_fields["info"] = "unknown"

    # Serialize fields to `k=v` pairs
    field_str = ", ".join(
        f"{k}={v}"
        for k, v in display_fields.items()
        if v not in (None, "", [], {})
    )

    if field_str:
        return f"({display_id} {{{field_str}}})"
    else:
        return f"({display_id})"
