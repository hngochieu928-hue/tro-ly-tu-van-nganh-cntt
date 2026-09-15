# ============================================================
# VISUALIZE_KG.PY — VẼ ĐỒ THỊ TRI THỨC BẰNG NETWORKX
# ============================================================
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from config import KG_DIR, BASE_DIR
from kg_schema import NODE_TYPES, RELATIONSHIP_TYPES


OUTPUT_PATH = os.path.join(BASE_DIR, "kg_visualization.png")


NODE_COLORS = {
    "University":         "#e74c3c",
    "Major":              "#3498db",
    "UniversityMajor":    "#9b59b6",
    "AdmissionCriteria":  "#f39c12",
    "SubjectCombination": "#2ecc71",
    "AdmissionMethod":    "#1abc9c",
    "MajorStatistic":     "#e67e22",
    "ProgramType":        "#34495e",
}


FILE_TO_NODE = {
    "kg_university.txt":          "University",
    "kg_major.txt":               "Major",
    "kg_university_major.txt":    "UniversityMajor",
    "kg_admission_criteria.txt":  "AdmissionCriteria",
    "kg_subject_combination.txt": "SubjectCombination",
    "kg_admission_method.txt":    "AdmissionMethod",
    "kg_major_statistic.txt":     "MajorStatistic",
    "kg_program_type.txt":        "ProgramType",
}


def count_kg_nodes():
    counts = {}
    if not os.path.exists(KG_DIR):
        return counts

    for filename in os.listdir(KG_DIR):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(KG_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            docs = [d for d in f.read().split("\n\n") if d.strip()]
        node_type = FILE_TO_NODE.get(filename, "Unknown")
        counts[node_type] = len(docs)
    return counts


def build_graph(counts):
    G = nx.DiGraph()

    for node_type in NODE_TYPES.keys():
        cnt = counts.get(node_type, 0)
        size = 1200 + min(cnt * 25, 4000)
        G.add_node(
            node_type,
            size=size,
            color=NODE_COLORS.get(node_type, "#95a5a6"),
            count=cnt,
            label=f"{node_type}\n({cnt})",
        )

    for rel, spec in RELATIONSHIP_TYPES.items():
        G.add_edge(spec["from"], spec["to"], label=rel)

    return G


def draw_graph(G, output_path=OUTPUT_PATH):
    plt.figure(figsize=(18, 13))
    plt.rcParams["font.family"] = "DejaVu Sans"

    pos = nx.circular_layout(G, scale=3.0)

    node_sizes = [G.nodes[n]["size"] for n in G.nodes()]
    node_colors = [G.nodes[n]["color"] for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.92,
        edgecolors="white",
        linewidths=2.5,
    )

    nx.draw_networkx_edges(
        G, pos,
        edge_color="#555555",
        arrows=True,
        arrowsize=25,
        width=2.0,
        alpha=0.65,
        connectionstyle="arc3,rad=0.15",
        min_source_margin=20,
        min_target_margin=25,
    )

    labels = {n: G.nodes[n]["label"] for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels,
        font_size=10, font_weight="bold",
        font_color="white",
    )

    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels,
        font_size=8, font_color="#222222",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.9),
    )

    plt.title(
        "Đồ thị tri thức miền tuyển sinh — Trường Đại học Điện lực\n"
        f"(8 loại nút, {G.number_of_edges()} loại quan hệ)",
        fontsize=15, fontweight="bold", pad=20,
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    print()
    print("=" * 68)
    print("   VẼ ĐỒ THỊ TRI THỨC")
    print("=" * 68)
    print()

    counts = count_kg_nodes()
    if not counts:
        print("⚠️  Chưa có dữ liệu KG.")
        print("   👉 Chạy: python build_kg.py")
        return

    print("📊 SỐ NÚT THEO LOẠI:")
    print("-" * 55)
    for node_type in NODE_TYPES.keys():
        cnt = counts.get(node_type, 0)
        mark = "✅" if cnt > 0 else "⚠️ "
        print(f"   {mark} {node_type:22s}: {cnt:5d} nút")
    print("-" * 55)

    G = build_graph(counts)
    print(f"   Tổng nút instance    : {sum(counts.values())}")
    print(f"   Tổng loại nút        : {G.number_of_nodes()}")
    print(f"   Tổng loại quan hệ    : {G.number_of_edges()}")
    print()

    print("🎨 Đang vẽ đồ thị...")
    output = draw_graph(G)
    print(f"✅ Đã lưu: {output}")
    print()
    print("👉 Mở file PNG để xem đồ thị.")
    print()


if __name__ == "__main__":
    main()