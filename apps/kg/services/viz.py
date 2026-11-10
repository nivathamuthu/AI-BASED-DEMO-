"""Interactive graph HTML with pyvis. A node click posts the node id to the parent page."""
from pyvis.network import Network

COLOURS = {
    "Aircraft": "#6b7385",
    "Component": "#1f5fbf",
    "Task": "#1b7f3b",
    "Regulation": "#b42318",
    "Procedure": "#7a4cc2",
    "Part": "#b26b00",
    "Interval": "#0e8a8a",
}

_CLICK = """
<script>
network.on("click", function (params) {
  if (params.nodes.length) { window.parent.postMessage({kgNode: params.nodes[0]}, "*"); }
});
</script>
"""


def graph_html(nodes: list[dict], edges: list[dict], types: set[str] | None = None) -> str:
    """Standalone HTML for the iframe. `types` keeps only nodes of those types and edges between them."""
    keep = [node for node in nodes if not types or node["type"] in types]
    ids = {node["id"] for node in keep}
    net = Network(height="620px", width="100%", directed=True, cdn_resources="in_line", bgcolor="#ffffff")
    for node in keep:
        sources = len(node.get("provenance", []))
        net.add_node(
            node["id"],
            label=node["name"],
            title=f"{node['type']} · {sources} source chunk(s)",
            color=COLOURS.get(node["type"], "#999999"),
            size=12 + min(sources, 10) * 2,
        )
    for edge in edges:
        if edge["source"] in ids and edge["target"] in ids:
            net.add_edge(edge["source"], edge["target"], label=edge["relation"], title=edge.get("evidence", "")[:300],
                         arrows="to", font={"size": 9, "align": "middle"})
    net.set_options('{"physics": {"barnesHut": {"springLength": 140}, "stabilization": {"iterations": 200}},'
                    ' "interaction": {"hover": true}}')
    html = net.generate_html(notebook=False)
    return html.replace("</body>", _CLICK + "</body>")
