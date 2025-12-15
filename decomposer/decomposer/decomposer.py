import pandas as pd
import pm4py
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import community as community_louvain
from dataclasses import dataclass
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pathlib import Path

def discover_dfg(path):
    df = pd.read_csv(path)
    df = pm4py.format_dataframe(df, case_id="Case ID", activity_key="Activity", timestamp_key="Start Date")
    event_log = pm4py.convert_to_event_log(df)
    dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)
    return dfg, start_activities, end_activities

@dataclass
class Community:
    activities: set[str]
    name: str

@dataclass
class DecompositionResult:
    partition: dict[str, int]
    communities: dict[int, Community]
    inter_edges: set[(tuple[str, str], int, str, str)]

Activity = str
Dfg = Counter[tuple[Activity, Activity]]
    
def decompose_dfg(dfg, start_activities, end_activities):
    if not isinstance(dfg, dict) or not dfg:
        raise ValueError("dfg must be a non-empty dict of (u,v)->weight")

    if community_louvain is None:
        raise ImportError("python-louvain is required. Install with: pip install python-louvain")

    G_dir = build_graph_from_dfg(dfg)
    G_undir = to_undirected_graph(G_dir)
    partition = community_louvain.best_partition(G_undir, weight="weight")
    communities_info = compute_communities(partition, G_undir)
    internal_edges, inter_edges = compute_edges(dfg, partition)

    return DecompositionResult(
        partition=partition,
        communities=communities_info,
        inter_edges=inter_edges
    )

def build_graph_from_dfg(dfg: Dfg) -> nx.DiGraph:
    G_dir = nx.DiGraph()
    for (u, v), w in dfg.items():
        if u is None or v is None:
            # Skip malformed entries
            continue
        G_dir.add_node(u)
        G_dir.add_node(v)
        # accumulate weights if multiple entries
        if G_dir.has_edge(u, v):
            G_dir[u][v]["weight"] += float(w)
        else:
            G_dir.add_edge(u, v, weight=float(w))
    return G_dir

def to_undirected_graph(G_dir: nx.DiGraph) -> nx.Graph:
    G_undir = nx.Graph()
    for u in G_dir.nodes:
        G_undir.add_node(u)

    # Sum weights in both directions
    for u, v in G_dir.edges:
        w_uv = G_dir[u][v].get("weight", 1.0)
        w_vu = G_dir[v][u].get("weight", 0.0) if G_dir.has_edge(v, u) else 0.0
        w_total = float(w_uv) + float(w_vu)
        if G_undir.has_edge(u, v):
            G_undir[u][v]["weight"] += w_total
        else:
            G_undir.add_edge(u, v, weight=w_total)
    return G_undir

def compute_communities(partition: dict[str, int], G_undir: nx.Graph) -> dict[int, Community]:
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, set()).add(node)

    # Subprocess naming: pick the node with highest degree (weighted) within each community
    communities_info = {}
    for cid, nodes in communities.items():
        subgraph = G_undir.subgraph(nodes)
        # weighted degree centrality heuristic
        degrees = {n: sum(d.get("weight", 1.0) for _, _, d in subgraph.edges(n, data=True)) for n in nodes}
        name = max(degrees, key=degrees.get) if degrees else f"community_{cid}"
        communities_info[cid] = Community(activities=set(nodes), name=name)

    return communities_info

def compute_edges(dfg: Dfg, partition: dict[str, int]):
    internal_edges = set()
    inter_edges = set()
    for (u, v), w in dfg.items():
        cu = partition.get(u)
        cv = partition.get(v)
        if cu is None or cv is None:
            continue
        if cu == cv:
            internal_edges.add(((u, v), float(w)))
        else:
            inter_edges.add(((u, v), float(w), cu, cv))
    return internal_edges, inter_edges

def save_workflow(decomposition_result: DecompositionResult, dfg: Dfg, start_activities, end_activities, name):
    out_dir = Path(f"reports/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_high_level(decomposition_result, out_dir)
    save_communities(decomposition_result, dfg, out_dir)
    pm4py.vis.save_vis_dfg(dfg, start_activities, end_activities, file_path = str(out_dir / f"sample_{name}.png"))

def save_high_level(decomposition_result: DecompositionResult, out_dir: Path):
    G = nx.Graph()
    communities = set(decomposition_result.partition.values())
    G.add_nodes_from(communities)
    edges = Counter()
    for (activity_edge, weight, community_u, community_v) in decomposition_result.inter_edges:
        edges[frozenset((community_u, community_v))] += weight
    for ((u, v), w) in edges.items():
        G.add_edge(u, v, weight=w)

    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    pos = nx.spring_layout(G, seed=42)  # Use a layout algorithm
    plt.figure()
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue"
    )
    nx.draw_networkx_edge_labels(
        G, 
        pos, 
        # with_labels=True, 
        edge_labels=edge_labels,
        # node_size=5000, 
        # node_color='lightblue',
        font_size=10, 
        # width=[G.edges[u, v]['weight'] / 100 for u, v in G.edges()]
    )
    plt.title("Decomposition Result: Flow Between Subprocesses")
    plt.savefig(out_dir / "high-level.png")

def save_communities(decomposition_result: DecompositionResult, dfg: Dfg, out_dir: Path):
    for community_id, community in decomposition_result.communities.items():
        G = nx.DiGraph()
        G.add_nodes_from(community.activities)
        # Add edges based on the DFG
        for (u, v), w in dfg.items():
            if u in community.activities and v in community.activities:
                G.add_edge(u, v, weight=w)
        # Save the community graph
        plt.figure()
        pos = nx.spring_layout(G, seed=42)  # Use a layout algorithm
        nx.draw(G, pos, with_labels=True)
        edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.savefig(out_dir / f"community_{community_id}.png")

def main():
    samples = [
        "logExample",
        "purchasingExample",
        "repairExample",
        "sepsisExample"
    ]
    for sample in samples:
        try:
            dfg, start_activities, end_activities = discover_dfg(f"/home/stachu/projects/process-mining-decomposition/data/{sample}.csv")
            result = decompose_dfg(dfg, start_activities, end_activities)
            save_workflow(result, dfg, start_activities, end_activities, sample)
        except Exception as e:
            print(f"Error processing {sample}: {e}")
