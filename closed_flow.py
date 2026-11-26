"""Verify feasible integral flows and their minimum-cut optimality witnesses."""

import re

DETAIL_KEYS = frozenset(
    (
        "valid_json",
        "flow_schema",
        "flow_capacity",
        "flow_conservation",
        "cut_partition_valid",
        "flow_cut_equal",
    )
)


def identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", value) is not None


def validate_expected(expected):
    if not isinstance(expected, dict) or set(expected) != {"nodes", "source", "sink", "arcs"}:
        return False
    nodes, arcs = expected["nodes"], expected["arcs"]
    if (
        not isinstance(nodes, list)
        or not 2 <= len(nodes) <= 48
        or not all(map(identifier, nodes))
        or len(set(nodes)) != len(nodes)
    ):
        return False
    if (
        expected["source"] not in nodes
        or expected["sink"] not in nodes
        or expected["source"] == expected["sink"]
    ):
        return False
    if not isinstance(arcs, dict) or not 1 <= len(arcs) <= 256 or not all(map(identifier, arcs)):
        return False
    return all(
        isinstance(edge, list)
        and len(edge) == 3
        and edge[0] in nodes
        and edge[1] in nodes
        and edge[0] != edge[1]
        and type(edge[2]) is int
        and 0 < edge[2] <= 10**6
        for edge in arcs.values()
    )


def score(item, reply):
    from hidden import strict_json

    network = item["expected"]
    try:
        cert = strict_json(reply["content"])
        parsed = True
    except (ValueError, TypeError, KeyError, RecursionError):
        cert, parsed = None, False
    shape = isinstance(cert, dict) and set(cert) == {"flow", "value", "source_side"}
    edges = network["arcs"]
    flows = cert["flow"] if shape else None
    value = cert["value"] if shape else None
    side = cert["source_side"] if shape else None
    flow_schema = (
        isinstance(flows, dict)
        and flows.keys() == edges.keys()
        and type(value) is int
        and all(type(amount) is int for amount in flows.values())
    )
    capacity = flow_schema and all(0 <= amount <= edges[label][2] for label, amount in flows.items())
    conserved = False
    if flow_schema:
        net_out = dict.fromkeys(network["nodes"], 0)
        for label, (tail, head, _) in edges.items():
            net_out[tail] += flows[label]
            net_out[head] -= flows[label]
        target = dict.fromkeys(network["nodes"], 0)
        target[network["source"]], target[network["sink"]] = value, -value
        conserved = net_out == target and value >= 0
    partition = (
        isinstance(side, list)
        and all(isinstance(node, str) and node in network["nodes"] for node in side)
        and len(set(side)) == len(side)
        and network["source"] in side
        and network["sink"] not in side
    )
    dual = False
    if partition and type(value) is int:
        inside = set(side)
        dual = value == sum(cap for u, v, cap in edges.values() if u in inside and v not in inside)
    detail = {"valid_json": float(parsed), "flow_schema": float(flow_schema),
              "flow_capacity": float(capacity), "flow_conservation": float(conserved),
              "cut_partition_valid": float(partition), "flow_cut_equal": float(dual)}
    correct = all(detail.values())
    return {"correct": correct, "score": float(correct), "detail": detail}
