"""
Organization hierarchy utility.

Builds a tree from a flat dict of team paths and provides query functions.
"""

from __future__ import annotations
from collections import defaultdict


class OrgTree:
    """Queryable organization tree built from a flat {team: path_list} dict."""

    def __init__(self, teams: dict[str, list[str]]):
        self._raw = teams
        self._tree: dict = {}
        self._children_of: dict[str, set[str]] = defaultdict(set)
        self._parent_of: dict[str, str] = {}
        self._all_nodes: set[str] = set()
        self._leaf_teams: set[str] = set(teams.keys())
        self._depth_of: dict[str, int] = {}

        self._build(teams)

    # ──────────────────────────────────────────────
    # Internal build
    # ──────────────────────────────────────────────

    def _build(self, teams: dict[str, list[str]]) -> None:
        for team, path in teams.items():
            node = self._tree
            for i, level in enumerate(path):
                self._all_nodes.add(level)
                if level not in self._depth_of:
                    self._depth_of[level] = i
                if level not in node:
                    node[level] = {}
                node = node[level]

                if i > 0:
                    parent = path[i - 1]
                    self._children_of[parent].add(level)
                    self._parent_of[level] = parent

    # ──────────────────────────────────────────────
    # Basic lookups
    # ──────────────────────────────────────────────

    def exists(self, name: str) -> bool:
        """Check if a node (team or division) exists in the tree."""
        return name in self._all_nodes

    def is_team(self, name: str) -> bool:
        """Check if a node is a leaf team (no children)."""
        return name in self._leaf_teams

    def is_division(self, name: str) -> bool:
        """Check if a node is a division (has children)."""
        return name in self._all_nodes and name not in self._leaf_teams

    def get_depth(self, name: str) -> int | None:
        """Get the depth level of a node (0 = root)."""
        return self._depth_of.get(name)

    # ──────────────────────────────────────────────
    # Parent / ancestor queries
    # ──────────────────────────────────────────────

    def get_parent(self, name: str) -> str | None:
        """Get the direct parent of a node."""
        return self._parent_of.get(name)

    def get_ancestors(self, name: str) -> list[str]:
        """Get all ancestors from direct parent up to root.

        Returns:
            List ordered from direct parent → root.
            e.g. for "Etch Team" → ["Fab Operations", "Manufacturing", "Company"]
        """
        ancestors = []
        current = name
        while current in self._parent_of:
            parent = self._parent_of[current]
            ancestors.append(parent)
            current = parent
        return ancestors

    def get_path(self, name: str) -> list[str] | None:
        """Get the full path from root to this node.

        Returns:
            e.g. ["Company", "Manufacturing", "Fab Operations", "Etch Team"]
            Returns None if the node doesn't exist.
        """
        if name not in self._all_nodes:
            return None
        # If it's a leaf team, return from raw data
        if name in self._raw:
            return list(self._raw[name])
        # Otherwise reconstruct from ancestors
        path = [name] + self.get_ancestors(name)
        path.reverse()
        return path

    # ──────────────────────────────────────────────
    # Children / descendant queries
    # ──────────────────────────────────────────────

    def get_children(self, name: str) -> list[str]:
        """Get direct children of a node.

        e.g. get_children("Manufacturing") → ["Fab Operations", "Quality", "Equipment"]
        """
        return sorted(self._children_of.get(name, set()))

    def get_all_descendants(self, name: str) -> list[str]:
        """Get ALL descendants (children, grandchildren, ...) recursively.

        Returns a flat sorted list of every node under the given node.
        """
        result = []
        stack = list(self._children_of.get(name, set()))
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(self._children_of.get(node, set()))
        return sorted(result)

    def get_teams_under(self, name: str) -> list[str]:
        """Get only leaf teams under a node (skips intermediate divisions).

        e.g. get_teams_under("Manufacturing")
             → ["CMP Team", "Etch Team", "Incoming QC", "Litho Team", ...]
        """
        return sorted(
            node for node in self.get_all_descendants(name)
            if node in self._leaf_teams
        )

    def get_divisions_under(self, name: str) -> list[str]:
        """Get only intermediate divisions under a node (skips leaf teams).

        e.g. get_divisions_under("Company")
             → ["Business", "Design", "Dev", "Equipment", "Fab Operations", ...]
        """
        return sorted(
            node for node in self.get_all_descendants(name)
            if node not in self._leaf_teams
        )

    def get_child_divisions(self, name: str) -> list[str]:
        """Get direct child divisions only (not leaf teams).

        e.g. get_child_divisions("IT") → ["Dev", "Infra", "Security"]
        """
        return sorted(
            child for child in self._children_of.get(name, set())
            if child not in self._leaf_teams
        )

    def get_child_teams(self, name: str) -> list[str]:
        """Get direct child teams only (leaf nodes).

        e.g. get_child_teams("Fab Operations") → ["CMP Team", "Etch Team", "Litho Team"]
        """
        return sorted(
            child for child in self._children_of.get(name, set())
            if child in self._leaf_teams
        )

    # ──────────────────────────────────────────────
    # Sibling queries
    # ──────────────────────────────────────────────

    def get_siblings(self, name: str) -> list[str]:
        """Get nodes at the same level under the same parent.

        e.g. get_siblings("Fab Operations") → ["Equipment", "Quality"]
             (excludes self)
        """
        parent = self._parent_of.get(name)
        if parent is None:
            return []
        return sorted(
            child for child in self._children_of[parent]
            if child != name
        )

    # ──────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────

    def search(self, keyword: str) -> list[str]:
        """Fuzzy search: find all nodes whose name contains the keyword (case-insensitive).

        e.g. search("team") → ["Backend Team", "Brand Team", "CMP Team", ...]
        """
        keyword_lower = keyword.lower()
        return sorted(
            node for node in self._all_nodes
            if keyword_lower in node.lower()
        )

    # ──────────────────────────────────────────────
    # Counting
    # ──────────────────────────────────────────────

    def count_teams_under(self, name: str) -> int:
        """Count leaf teams under a node."""
        return len(self.get_teams_under(name))

    def count_all_under(self, name: str) -> int:
        """Count all descendants (divisions + teams) under a node."""
        return len(self.get_all_descendants(name))

    # ──────────────────────────────────────────────
    # Tree visualization
    # ──────────────────────────────────────────────

    def print_tree(self, root: str | None = None, indent: int = 0) -> str:
        """Get a visual tree string starting from a given root.

        Args:
            root: Starting node. If None, prints from the top-level root(s).
            indent: Internal use for recursion depth.

        Returns:
            A string with the formatted tree.
        """
        lines: list[str] = []

        if root is None:
            # Find root nodes (nodes with no parent)
            roots = sorted(
                node for node in self._all_nodes
                if node not in self._parent_of
            )
            for r in roots:
                lines.append(self.print_tree(r, indent))
            return "\n".join(lines)

        prefix = "    " * indent
        marker = "📁" if self.is_division(root) else "👥"
        team_count = self.count_teams_under(root)
        suffix = f"  ({team_count} teams)" if self.is_division(root) else ""
        lines.append(f"{prefix}{marker} {root}{suffix}")

        for child in self.get_children(root):
            lines.append(self.print_tree(child, indent + 1))

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # Common ancestor
    # ──────────────────────────────────────────────

    def get_common_ancestor(self, name_a: str, name_b: str) -> str | None:
        """Find the lowest common ancestor of two nodes.

        e.g. get_common_ancestor("Etch Team", "Incoming QC") → "Manufacturing"
        """
        path_a = self.get_path(name_a)
        path_b = self.get_path(name_b)
        if path_a is None or path_b is None:
            return None

        common = None
        for a, b in zip(path_a, path_b):
            if a == b:
                common = a
            else:
                break
        return common

    # ──────────────────────────────────────────────
    # Level-based queries
    # ──────────────────────────────────────────────

    def get_nodes_at_depth(self, depth: int) -> list[str]:
        """Get all nodes at a specific depth level.

        e.g. get_nodes_at_depth(1) → ["Business", "IT", "Manufacturing", "R&D"]
        """
        return sorted(
            node for node, d in self._depth_of.items()
            if d == depth
        )

    def get_subtree_dict(self, name: str) -> dict | None:
        """Get the subtree as a nested dict (useful for JSON export).

        Returns nested dict like:
            {"Fab Operations": {"Etch Team": {}, "Litho Team": {}, "CMP Team": {}}}
        """
        if name not in self._all_nodes:
            return None

        def _build_subtree(node_name: str) -> dict:
            children = self.get_children(node_name)
            if not children:
                return {}
            return {child: _build_subtree(child) for child in children}

        return {name: _build_subtree(name)}
