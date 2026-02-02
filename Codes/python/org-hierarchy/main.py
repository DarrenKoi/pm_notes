"""Demo usage of OrgTree with sample data."""

from sample_data import TEAMS
from org_tree import OrgTree


def main():
    org = OrgTree(TEAMS)

    print("=" * 60)
    print("FULL TREE")
    print("=" * 60)
    print(org.print_tree())

    print("\n" + "=" * 60)
    print("QUERY EXAMPLES")
    print("=" * 60)

    # 1. What teams are under Manufacturing?
    print("\n--- Teams under 'Manufacturing' ---")
    for t in org.get_teams_under("Manufacturing"):
        print(f"  {t}")

    # 2. What divisions are under Company?
    print("\n--- Divisions under 'Company' ---")
    for d in org.get_divisions_under("Company"):
        print(f"  {d}")

    # 3. Direct children of IT
    print("\n--- Direct children of 'IT' ---")
    for c in org.get_children("IT"):
        print(f"  {c}")

    # 4. Path of a team
    print("\n--- Path of 'Etch Team' ---")
    print(f"  {' → '.join(org.get_path('Etch Team'))}")

    # 5. Parent lookup
    print("\n--- Parent of 'Fab Operations' ---")
    print(f"  {org.get_parent('Fab Operations')}")

    # 6. Ancestors
    print("\n--- Ancestors of 'Backend Team' ---")
    print(f"  {org.get_ancestors('Backend Team')}")

    # 7. Siblings
    print("\n--- Siblings of 'Fab Operations' ---")
    for s in org.get_siblings("Fab Operations"):
        print(f"  {s}")

    # 8. Search
    print("\n--- Search 'team' ---")
    for r in org.search("team"):
        print(f"  {r}")

    # 9. Common ancestor
    print("\n--- Common ancestor of 'Etch Team' and 'Incoming QC' ---")
    print(f"  {org.get_common_ancestor('Etch Team', 'Incoming QC')}")

    # 10. Count
    print("\n--- Team counts ---")
    print(f"  Under Company: {org.count_teams_under('Company')}")
    print(f"  Under IT: {org.count_teams_under('IT')}")
    print(f"  Under R&D: {org.count_teams_under('R&D')}")

    # 11. Nodes at depth
    print("\n--- Nodes at depth 1 (top divisions) ---")
    for n in org.get_nodes_at_depth(1):
        print(f"  {n}")

    # 12. Subtree as dict
    print("\n--- Subtree of 'IT' (dict) ---")
    import json
    print(json.dumps(org.get_subtree_dict("IT"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
