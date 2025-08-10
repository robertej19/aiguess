# Generate a 500-employee example org CSV where some parents have up to 15 children
import pandas as pd
import random
from pathlib import Path

random.seed(7)

# --- name generation ---
FIRST_NAMES = [
    "Alex","Taylor","Jordan","Casey","Riley","Avery","Morgan","Quinn","Jamie","Cameron",
    "Drew","Parker","Devon","Logan","Rowan","Emery","Elliot","Reese","Sasha","Skye",
    "Kai","Remy","Micah","Hayden","Blake","Shawn","Kendall","Harley","Sidney","Shannon",
    "Noah","Olivia","Liam","Emma","Ava","Sophia","Isabella","Mia","Amelia","Harper",
    "Ethan","Lucas","Mason","Levi","Benjamin","Elijah","Henry","Sebastian","Jack","Daniel",
    "Chloe","Lily","Ella","Grace","Zoe","Nora","Scarlett","Layla","Aria","Mila",
    "Mateo","Santiago","Mateus","Diego","Andrés","Sofia","Valentina","Camila","Lucia","Elena",
    "Priya","Ananya","Aarav","Raj","Arjun","Isha","Neha","Amit","Rohan","Kiran",
    "Hiro","Yuki","Sora","Ren","Aiko","Mei","Jin","Min","Hana","Suki"
]

LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
    "Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin",
    "Lee","Perez","Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson",
    "Walker","Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
    "Green","Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell","Carter","Roberts",
    "Khan","Singh","Patel","Shah","Kumar","Gupta","Das","Iyer","Chowdhury","Bose",
    "Yamamoto","Sato","Kobayashi","Tanaka","Watanabe","Suzuki","Nakamura","Kim","Park","Choi",
    "Chen","Wang","Zhang","Liu","Lin","Huang","Chang","Wu","Li","Zhao"
]

def unique_name(used):
    for _ in range(20000):
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        if name not in used:
            used.add(name)
            return name
    # fallback
    i = 1
    while True:
        name = f"Employee {i}"
        if name not in used:
            used.add(name)
            return name
        i += 1

# --- role pools by team ---
ROLE_POOLS = {
    "Software": ["SWE 1","SWE 2","SWE 3","QA Engineer","DevOps Engineer","SRE"],
    "Data": ["Data Scientist 1","Data Scientist 2","Data Scientist 3","Data Engineer 1","Data Engineer 2","ML Engineer 1","ML Engineer 2","Analytics Engineer"],
    "Platform": ["Platform Engineer","Site Reliability Engineer","DevOps Engineer","Cloud Engineer"],
    "Infrastructure": ["Infrastructure Engineer","Systems Engineer","Network Engineer","SRE"],
    "Hardware": ["Hardware Engineer","Electrical Engineer","Mechanical Engineer","Firmware Engineer","Test Engineer"],
    "Product": ["Product Manager 1","Product Manager 2","Product Manager 3","Program Manager"],
    "Design": ["UX Designer","UI Designer","Product Designer","Design Researcher"],
    "Sales": ["Account Executive","Sales Manager","Sales Development Rep","Solutions Engineer"],
    "Marketing": ["Marketing Manager","Content Strategist","SEO Specialist","Growth Marketer","Demand Gen Manager"],
    "Finance": ["Financial Analyst","Senior Financial Analyst","FP&A Manager","Payroll Specialist"],
    "Legal": ["Legal Counsel","Paralegal","Contracts Manager"],
    "HR": ["HR Generalist","Recruiter","People Ops Manager","Comp & Benefits Analyst"],
    "Operations": ["Operations Manager","Business Operations Analyst","Workplace Coordinator"],
    "IT": ["IT Support Specialist","Systems Administrator","Network Administrator","IT Analyst"],
    "Security": ["Security Engineer","Security Analyst","GRC Analyst","Application Security Engineer"],
    "Customer Success": ["Customer Success Manager","Support Specialist","Technical Support Engineer","Implementation Specialist"],
    "Support": ["Support Specialist","Technical Support Engineer","Escalations Engineer"],
    "Supply Chain": ["Supply Chain Analyst","Procurement Specialist","Buyer","Inventory Analyst"],
    "Manufacturing": ["Manufacturing Engineer","Production Supervisor","Quality Engineer","Process Technician"],
    "Tax": ["Tax Analyst","Senior Tax Analyst","Tax Manager"],
    "Management": ["Executive"],
}

# CxO/VP layer under CEO (manager = "CEO")
LEADERS = [
    ("COO","Operations"),
    ("CFO","Finance"),
    ("CTO","Software"),
    ("CPO","Product"),
    ("CMO","Marketing"),
    ("CHRO","HR"),
    ("CIO","IT"),
    ("GC","Legal"),
    ("CISO","Security"),
    ("CRO","Sales"),
    ("SVP Supply Chain","Supply Chain"),
    ("SVP Manufacturing","Manufacturing"),
    ("SVP Customer","Customer Success"),
    ("VP Hardware","Hardware"),
    ("VP Data","Data"),
    ("VP Platform","Platform"),
    ("VP Infrastructure","Infrastructure"),
    ("VP Support","Support"),
    ("VP Design","Design"),
    ("VP Tax","Tax"),
]

rows = []
used_names = set()
children_count = {}  # parent -> number of direct children

def add_row(employee, manager, role, team):
    rows.append({
        "employee": employee,
        "manager": manager if manager is not None else "",
        "role": role,
        "team": team
    })
    if manager:
        children_count[manager] = children_count.get(manager, 0) + 1

# CEO
add_row("CEO", "", "Executive", "Management")

# Leaders
for title, team in LEADERS:
    add_row(title, "CEO", title, team)

target_total = 500

def grow_team(parent, team):
    # directors (1-3)
    directors = []
    for _ in range(random.randint(1,3)):
        name = unique_name(used_names)
        add_row(name, parent, f"Director of {team}", team)
        directors.append(name)
        if len(rows) >= target_total: 
            return
    # managers (2-6) per director
    managers = []
    for d in directors:
        for _ in range(random.randint(2,6)):
            name = unique_name(used_names)
            add_row(name, d, f"{team} Manager", team)
            managers.append(name)
            if len(rows) >= target_total:
                return
    # ICs (4-10) per manager
    pool = ROLE_POOLS.get(team, ["Associate"])
    for m in managers:
        for _ in range(random.randint(4,10)):
            name = unique_name(used_names)
            role = random.choice(pool)
            add_row(name, m, role, team)
            if len(rows) >= target_total:
                return

# First pass growth for each leader
for title, team in LEADERS:
    if len(rows) >= target_total: break
    grow_team(title, team)

# Identify managers/directors to widen (give many direct ICs up to 15)
def team_of(emp):
    for r in rows:
        if r["employee"] == emp:
            return r["team"]
    return "Operations"

# Collect potential parents to widen: managers first, then directors
potential_parents = [r["employee"] for r in rows if "Manager" in r["role"]]
directors = [r["employee"] for r in rows if "Director" in r["role"]]
potential_parents += directors

random.shuffle(potential_parents)

# Target number of wide parents
wide_targets = min(25, max(10, len(potential_parents)//10))  # ~10% but between 10 and 25
wide_parents = potential_parents[:wide_targets]

for p in wide_parents:
    desired = random.randint(12,15)  # up to 15 children
    cur = children_count.get(p, 0)
    team = team_of(p)
    pool = ROLE_POOLS.get(team, ["Associate"])
    # add ICs under p until desired or target_total reached
    while cur < desired and len(rows) < target_total:
        name = unique_name(used_names)
        role = random.choice(pool)
        add_row(name, p, role, team)
        cur += 1

# If still short, fill under random managers/directors without exceeding 15 children each
parents_fill = potential_parents + [r["employee"] for r in rows if r["role"] in ("CTO","CFO","COO","CIO","CPO","CMO","GC","CISO","CRO")]
idx = 0
while len(rows) < target_total and parents_fill:
    p = parents_fill[idx % len(parents_fill)]
    idx += 1
    cur = children_count.get(p, 0)
    if cur >= 15:
        continue
    team = team_of(p)
    pool = ROLE_POOLS.get(team, ["Associate"])
    name = unique_name(used_names)
    role = random.choice(pool)
    add_row(name, p, role, team)

# Build DataFrame
df = pd.DataFrame(rows, columns=["employee","manager","role","team"])

# Sanity checks
assert df["employee"].is_unique, "Employee names are not unique"
assert df.loc[df["manager"].eq(""), "employee"].tolist() == ["CEO"], "There should be exactly one CEO with blank manager"
assert len(df) == 500, f"Expected 500 rows, got {len(df)}"

# Save files
out1 = Path("sample_250.csv")
out2 = Path("sample_250.csv")  # overwrite to make the app pick it up
df.to_csv(out1, index=False)
df.to_csv(out2, index=False)

# quick report of max direct reports
from collections import Counter, defaultdict
parent_counts = Counter(df["manager"][df["manager"] != ""])
max_children = parent_counts.most_common(5)

