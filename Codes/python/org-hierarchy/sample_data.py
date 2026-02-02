"""
Sample organization hierarchy data.

Structure: { "team_name": ["level1", "level2", ..., "team_name"] }
- The list represents the path from the top-most group down to the team itself.
- The last element is always the team itself.

Example hierarchy:
    Company
    ├── Manufacturing
    │   ├── Fab Operations
    │   │   ├── Etch Team
    │   │   ├── Litho Team
    │   │   └── CMP Team
    │   ├── Quality
    │   │   ├── Incoming QC
    │   │   └── Outgoing QC
    │   └── Equipment
    │       ├── Maintenance Alpha
    │       └── Maintenance Beta
    ├── R&D
    │   ├── Process Dev
    │   │   ├── Advanced Node
    │   │   └── Legacy Node
    │   ├── Design
    │   │   ├── Circuit Design
    │   │   └── Layout Team
    │   └── Test Engineering
    │       ├── Reliability
    │       └── Failure Analysis
    ├── IT
    │   ├── Infra
    │   │   ├── Cloud Ops
    │   │   └── Network Team
    │   ├── Dev
    │   │   ├── Backend Team
    │   │   ├── Frontend Team
    │   │   └── Data Platform
    │   └── Security
    │       └── SecOps
    └── Business
        ├── Sales
        │   ├── Domestic Sales
        │   └── Global Sales
        ├── Marketing
        │   ├── Brand Team
        │   └── Digital Marketing
        └── Finance
            ├── Accounting
            └── FP&A
"""

TEAMS: dict[str, list[str]] = {
    # Manufacturing > Fab Operations
    "Etch Team": ["Company", "Manufacturing", "Fab Operations", "Etch Team"],
    "Litho Team": ["Company", "Manufacturing", "Fab Operations", "Litho Team"],
    "CMP Team": ["Company", "Manufacturing", "Fab Operations", "CMP Team"],
    # Manufacturing > Quality
    "Incoming QC": ["Company", "Manufacturing", "Quality", "Incoming QC"],
    "Outgoing QC": ["Company", "Manufacturing", "Quality", "Outgoing QC"],
    # Manufacturing > Equipment
    "Maintenance Alpha": ["Company", "Manufacturing", "Equipment", "Maintenance Alpha"],
    "Maintenance Beta": ["Company", "Manufacturing", "Equipment", "Maintenance Beta"],

    # R&D > Process Dev
    "Advanced Node": ["Company", "R&D", "Process Dev", "Advanced Node"],
    "Legacy Node": ["Company", "R&D", "Process Dev", "Legacy Node"],
    # R&D > Design
    "Circuit Design": ["Company", "R&D", "Design", "Circuit Design"],
    "Layout Team": ["Company", "R&D", "Design", "Layout Team"],
    # R&D > Test Engineering
    "Reliability": ["Company", "R&D", "Test Engineering", "Reliability"],
    "Failure Analysis": ["Company", "R&D", "Test Engineering", "Failure Analysis"],

    # IT > Infra
    "Cloud Ops": ["Company", "IT", "Infra", "Cloud Ops"],
    "Network Team": ["Company", "IT", "Infra", "Network Team"],
    # IT > Dev
    "Backend Team": ["Company", "IT", "Dev", "Backend Team"],
    "Frontend Team": ["Company", "IT", "Dev", "Frontend Team"],
    "Data Platform": ["Company", "IT", "Dev", "Data Platform"],
    # IT > Security
    "SecOps": ["Company", "IT", "Security", "SecOps"],

    # Business > Sales
    "Domestic Sales": ["Company", "Business", "Sales", "Domestic Sales"],
    "Global Sales": ["Company", "Business", "Sales", "Global Sales"],
    # Business > Marketing
    "Brand Team": ["Company", "Business", "Marketing", "Brand Team"],
    "Digital Marketing": ["Company", "Business", "Marketing", "Digital Marketing"],
    # Business > Finance
    "Accounting": ["Company", "Business", "Finance", "Accounting"],
    "FP&A": ["Company", "Business", "Finance", "FP&A"],
}
