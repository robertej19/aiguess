import base64
import io
import math
import re
from collections import defaultdict, deque

import dash
from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objs as go
from plotly.colors import sequential as pseq

# -------------------------
# CSV cleaning (trim trailing whitespace in every textual cell & column name)
# -------------------------

def clean_trailing_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Remove trailing whitespace (including NBSP) from textual cells and column names.
    Leaves non-text dtypes unchanged and preserves NaN/None.
    """
    dfc = df.copy()

    # Clean column names (right-trim)
    new_cols = {}
    for col in dfc.columns:
        new_name = re.sub(r"[\s\u00A0]+$", "", str(col))
        if new_name != col:
            new_cols[col] = new_name
    if new_cols:
        dfc = dfc.rename(columns=new_cols)

    # Clean cell values for text-like columns only
    for col in dfc.columns:
        if pd.api.types.is_string_dtype(dfc[col]) or dfc[col].dtype == object:
            dfc[col] = dfc[col].map(lambda x: x if pd.isna(x) else re.sub(r"[\s\u00A0]+$", "", str(x)))
    return dfc


# -------------------------
# Org tree core
# -------------------------
class OrgTree:
    def __init__(self, ceo):
        self.root = ceo
        self.children = defaultdict(list)   # manager -> [direct reports]
        self.parent = {}                    # employee -> manager
        self.nodes = set([ceo])

    def add_report(self, manager, employee):
        if employee in self.parent:
            raise ValueError(f"{employee} already has a manager: {self.parent[employee]}")
        if employee == self.root:
            raise ValueError("The CEO cannot report to someone else.")
        self.children[manager].append(employee)
        self.parent[employee] = manager
        self.nodes.update([manager, employee])

    def depths(self) -> dict:
        d = {self.root: 0}
        q = deque([self.root])
        while q:
            n = q.popleft()
            for c in self.children.get(n, []):
                d[c] = d[n] + 1
                q.append(c)
        return d


# -------------------------
# Build from DataFrame
# -------------------------

def build_tree_from_df(df: pd.DataFrame) -> OrgTree:
    # normalize column names (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    if "employee" not in cols or "manager" not in cols:
        raise ValueError("CSV must contain columns: employee, manager")
    df = df.rename(columns={cols["employee"]: "employee", cols["manager"]: "manager"})

    # Robust null/trim handling for CEO detection
    def _to_null_or_str(x):
        if pd.isna(x) or x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return s

    def _to_str(x):
        return str(x).strip()

    df["employee"] = df["employee"].apply(_to_str)
    df["manager"] = df["manager"].apply(_to_null_or_str)

    # find CEO (manager is None)
    roots = df[df["manager"].isna()]["employee"].unique().tolist()
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one CEO (row with blank/None manager). Found: {roots}")
    ceo = roots[0]

    tree = OrgTree(ceo)
    # Add all relationships
    for _, row in df.dropna(subset=["manager"]).iterrows():
        m = row["manager"]
        e = row["employee"]
        if m == e:
            raise ValueError(f"{e} cannot manage themselves")
        tree.add_report(m, e)

    # Validate that every non-CEO appears exactly once in employee column
    employees = set(df["employee"].tolist())
    managers = set([_to_null_or_str(x) for x in df["manager"].tolist() if _to_null_or_str(x) is not None])
    unknown_managers = managers - employees - {ceo}
    if unknown_managers:
        raise ValueError(f"Manager(s) not listed as employee(s): {sorted(unknown_managers)}")

    return tree


# -------------------------
# Helpers for layout
# -------------------------

def _subtree_sizes(tree: OrgTree) -> dict:
    """Return dict of node -> number of nodes in its subtree (including itself)."""
    sizes = {}

    def dfs(n):
        total = 1
        for c in tree.children.get(n, []):
            total += dfs(c)
        sizes[n] = total
        return total

    dfs(tree.root)
    return sizes


def _spread_indices_linear(k: int) -> list:
    """Return k indices [0..k-1] arranged to maximize spacing of early items.
    Strategy: repeatedly place at the midpoint of the currently largest gap.
    This spreads the first (heaviest) items far apart along a line segment.
    """
    if k <= 1:
        return [0]
    intervals = [(0, k - 1)]  # inclusive index intervals
    order = []
    while len(order) < k:
        # pick interval with greatest length
        i = max(range(len(intervals)), key=lambda j: intervals[j][1] - intervals[j][0])
        a, b = intervals.pop(i)
        if a > b:
            continue
        mid = (a + b) // 2
        order.append(mid)
        # push left and right leftover intervals
        if mid - 1 >= a:
            intervals.append((a, mid - 1))
        if mid + 1 <= b:
            intervals.append((mid + 1, b))
    return order


def _order_children_spread(kids: list, weight_lookup: dict) -> list:
    """Order children so that heavy subtrees are maximally separated.
    - kids are sorted by weight (desc)
    - positions are chosen by _spread_indices_linear
    """
    if len(kids) <= 2:
        return kids[:]  # nothing fancy needed
    # sort children by subtree size (excluding the node itself to emphasize descendants)
    ranked = sorted(kids, key=lambda c: (weight_lookup.get(c, 1) - 1), reverse=True)
    slots = _spread_indices_linear(len(kids))
    arranged = [None] * len(kids)
    for child, idx in zip(ranked, slots):
        arranged[idx] = child
    # fill any Nones with remaining kids in stable order
    rem = [c for c in kids if c not in arranged]
    for i in range(len(arranged)):
        if arranged[i] is None:
            arranged[i] = rem.pop(0)
    return arranged


# -------------------------
# RULE-BASED RADIAL LAYOUT (fixed rings)
# -------------------------
# Rule 1: All nodes at the same original degree (distance from CEO in the FULL org)
#         sit on the same radius, **unchanged** across filtering/recalcs.
# Rule 2: Siblings divide their parent's angular sector equally, but we **reorder**
#         siblings to spread heavy subtrees apart (maximize spacing of large branches).

def radial_layout_rules(tree: OrgTree, radius_step=1.0, depth_lookup: dict | None = None):
    pos = {}
    sizes = _subtree_sizes(tree)

    def place(node, depth, a0, a1):
        theta = 0.5 * (a0 + a1)
        # Use fixed ring from depth_lookup if provided; fallback to current depth
        d_ring = depth_lookup.get(node, depth) if depth_lookup else depth
        r = d_ring * radius_step
        pos[node] = (r * math.cos(theta), r * math.sin(theta))

        kids = tree.children.get(node, [])
        if not kids:
            return

        # Reorder children so heavy subtrees are separated
        ordered_kids = _order_children_spread(kids, sizes)
        k = len(ordered_kids)
        span = (a1 - a0) / max(k, 1)
        for i, c in enumerate(ordered_kids):
            child_a0 = a0 + i * span
            child_a1 = a0 + (i + 1) * span
            place(c, depth + 1, child_a0, child_a1)

    # root owns the whole circle
    place(tree.root, 0, 0.0, 2 * math.pi)
    return pos


# -------------------------
# Edge geometry (piecewise: straight trunk then outward curve)
# -------------------------

def _quad_bezier_points(p0, p1, pc, steps=28):
    """Sample a quadratic Bezier from p0->p1 with control point pc."""
    xs, ys = [], []
    for j in range(steps + 1):
        t = j / steps
        mt = 1 - t
        x = mt * mt * p0[0] + 2 * mt * t * pc[0] + t * t * p1[0]
        y = mt * mt * p0[1] + 2 * mt * t * pc[1] + t * t * p1[1]
        xs.append(x)
        ys.append(y)
    xs.append(None)
    ys.append(None)
    return xs, ys


def edge_coordinates_with_curves(pos: dict, tree: OrgTree, radius_step: float = 1.0):
    """Return x,y lists for edges.
    For any non-CEO node with >1 children (i.e., branches), each edge is:
      1) a straight **trunk** from the parent radius R to mid-radius R+(r/2) along the
         parent's radial direction, then
      2) an **outward-bending** quadratic Bezier from that midpoint to the child at
         radius R+r.
    CEO edges and single-child edges remain straight.
    """
    ex, ey = [], []

    outward_eps = 0.06 * radius_step  # small push so the curve bulges outward

    for manager, kids in tree.children.items():
        if not kids:
            continue
        x0, y0 = pos[manager]
        r0 = math.hypot(x0, y0)
        th0 = math.atan2(y0, x0)

        if manager != tree.root and len(kids) > 1:
            # Compute a single trunk midpoint for this parent (drawn once)
            # Use the farthest child radius to define a consistent midpoint trunk.
            r_children = [math.hypot(pos[c][0], pos[c][1]) for c in kids]
            r1_max = max(r_children)
            r_mid = 0.5 * (r0 + r1_max)
            mx = r_mid * math.cos(th0)
            my = r_mid * math.sin(th0)

            # Draw the straight trunk once
            ex += [x0, mx, None]
            ey += [y0, my, None]

            # Now, for each child, draw the outward curve from the midpoint to the child
            for child in kids:
                x1, y1 = pos[child]
                r1 = math.hypot(x1, y1)
                th1 = math.atan2(y1, x1)

                # Control point at child's angle but slightly beyond child's radius
                r_ctrl = r1 + outward_eps
                cx = r_ctrl * math.cos(th1)
                cy = r_ctrl * math.sin(th1)

                xs, ys = _quad_bezier_points((mx, my), (x1, y1), (cx, cy), steps=28)
                ex += xs
                ey += ys
        else:
            # straight lines (CEO or single-child)
            for child in kids:
                x1, y1 = pos[child]
                ex += [x0, x1, None]
                ey += [y0, y1, None]

    return ex, ey


# -------------------------
# Color helpers — Viridis with semantic grouping
# -------------------------
VIRIDIS = pseq.Viridis  # list of hex strings along the viridis colormap

ROLE_GROUPS = [
    # 0 Tech / Eng / Data (keep DS & SWE close)
    [
        'engineer','developer','swe','software','data','data scientist','ml','ai','scientist',
        'analytics','platform','devops','infra','infrastructure','sre','backend','frontend',
        'full stack','qa','test','security','research engineer'
    ],
    # 1 Product / Design / PM
    ['product','pm','program','project','designer','design','ux','ui'],
    # 2 Hardware / Elec / Mech / Firmware / Mfg
    ['hardware','mechanical','electrical','firmware','manufacturing','industrial','mechatronics'],
    # 3 Sales / Marketing / GTM / CS
    ['sales','account','bd','business development','marketing','growth','customer','support','success','partnership','csm','solutions','presales','pre-sales'],
    # 4 People / HR / Recruiting
    ['hr','people','recruit','talent','compensation','benefits','payroll'],
    # 5 Finance / Legal / Tax / Risk
    ['finance','accounting','fp&a','legal','counsel','tax','audit','compliance','risk','treasury'],
    # 6 Operations / IT / Facilities / Supply
    ['operations','ops','supply','logistics','facilities','it','sysadmin','office','procurement','security operations'],
    # 7 Leadership / Executive
    ['chief','ceo','cto','cfo','coo','cso','cio','ciso','executive','vp','svp','evp','director','head','lead'],
    # 8 Science (non-data)
    ['research','biology','chemistry','physics','lab']
]

TEAM_GROUPS = [
    # 0 Software / Data / Platform / Infra (close together)
    ['software','data','ml','ai','platform','infra','infrastructure','backend','frontend','devops','sre','security','qa','testing'],
    # 1 Hardware & related
    ['hardware','firmware','electrical','mechanical','manufacturing','industrial','lab'],
    # 2 Product / Design
    ['product','design','ux','ui','research'],
    # 3 Sales / Marketing / GTM / CS
    ['sales','marketing','growth','gtm','business development','bd','success','support','customer','solutions'],
    # 4 People / HR
    ['hr','people','recruiting','talent'],
    # 5 Finance / Legal / Tax
    ['finance','accounting','legal','tax','compliance','audit','risk'],
    # 6 Operations / IT
    ['operations','ops','it','facilities','logistics','supply','procurement'],
    # 7 Management / Exec
    ['management','executive','leadership']
]


def _classify_group(value: str, attr: str) -> int:
    s = (value or '').lower()
    groups = ROLE_GROUPS if attr == 'role' else TEAM_GROUPS
    for gi, keywords in enumerate(groups):
        for kw in keywords:
            if kw in s:
                return gi
    return len(groups)  # "other"


def _sample_viridis(t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    n = len(VIRIDIS)
    idx = int(round(t * (n - 1)))
    return VIRIDIS[idx]


def _viridis_category_colors(categories: list[str], attr: str) -> dict:
    # Separate out 'Unspecified'
    cats = [c for c in categories if c != 'Unspecified']
    if not cats:
        return {'Unspecified': '#BDBDBD'}

    # Group present categories
    present_groups = defaultdict(list)
    for c in cats:
        gi = _classify_group(c, attr)
        present_groups[gi].append(c)

    # Sort groups and their members for stability
    grouped = [(gi, sorted(members)) for gi, members in sorted(present_groups.items(), key=lambda x: x[0])]
    g = len(grouped)

    # Allocate disjoint spans across [0,1]
    pad = 0.04  # padding at both ends
    total_width = 1.0 - 2 * pad
    group_width = total_width / max(g, 1)
    inner_margin = 0.18  # leave space between groups
    inner_width = group_width * (1 - inner_margin)

    cmap = {}
    for i, (gi, members) in enumerate(grouped):
        start = pad + i * group_width + (group_width - inner_width) / 2
        k = len(members)
        if k == 1:
            ts = [start + inner_width * 0.5]
        else:
            ts = [start + inner_width * ((j + 0.5) / k) for j in range(k)]
        for c, t in zip(members, ts):
            cmap[c] = _sample_viridis(t)

    # Unspecified as neutral gray
    if 'Unspecified' in categories:
        cmap['Unspecified'] = '#BDBDBD'
    return cmap


# -------------------------
# Node size helpers — bigger near center, smaller outward
# -------------------------

def compute_node_sizes(pos: dict[str, tuple[float,float]], min_size: float = 7.0, max_size: float = 22.0) -> dict:
    """Map each node to a marker size that decreases linearly with radius.
    Nodes on the CEO ring (r≈0) get max_size; farthest ring gets min_size.
    """
    if not pos:
        return {}
    radii = [math.hypot(x, y) for (x, y) in pos.values()]
    rmax = max(radii) if radii else 1.0
    sizes = {}
    for n, (x, y) in pos.items():
        r = math.hypot(x, y)
        t = 0.0 if rmax == 0 else (r / rmax)
        sizes[n] = max_size - (max_size - min_size) * t
    return sizes


# -------------------------
# Plotly figure builder
# -------------------------

def org_to_figure(tree: OrgTree, pos: dict, df: pd.DataFrame, color_by: str | None, min_size: float, max_size: float):
    # Edge trace (draw **first** so it stays behind nodes)
    ex, ey = edge_coordinates_with_curves(pos, tree)
    edge_trace = go.Scatter(
        x=ex, y=ey, mode='lines', line=dict(width=1, color='#888'),
        hoverinfo='skip', name='edges', showlegend=False
    )

    # Build metadata lookup (use plain dicts to avoid pandas Series truthiness issues)
    df_local = df.copy()
    if 'employee' not in df_local.columns:
        raise ValueError("Data must include an 'employee' column for coloring.")
    df_local['employee'] = df_local['employee'].astype(str)
    meta = df_local.set_index('employee').to_dict(orient='index')  # {employee: {col: val, ...}}

    def hover_for(n: str):
        row = meta.get(str(n), {})
        role = row.get('role')
        team = row.get('team')
        parts = [f"{n}"]
        if role is not None and str(role).strip():
            parts.append(f"Role: {role}")
        if team is not None and str(team).strip():
            parts.append(f"Team: {team}")
        return "<br>".join(parts)

    # Decide attribute for coloring
    attr = None
    if color_by and color_by.lower() in ['role','team'] and color_by in df_local.columns:
        attr = color_by

    # Compute per-node sizes from radius using UI-controlled range
    size_map = compute_node_sizes(pos, min_size=min_size, max_size=max_size)

    # If no valid attribute, single node trace (NO labels; hover only)
    if attr is None:
        nodes_all = list(tree.nodes)
        xs = [pos[n][0] for n in nodes_all]
        ys = [pos[n][1] for n in nodes_all]
        hovers = [hover_for(n) for n in nodes_all]
        sizes = [size_map.get(n, 10.0) for n in nodes_all]
        node_trace = go.Scatter(
            x=xs, y=ys, mode='markers', marker=dict(size=sizes),
            hoverinfo='text', text=hovers, name='nodes', showlegend=False,
            customdata=nodes_all,
        )
        title = 'Radial Organization Chart (rule-based)'
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=title, showlegend=False,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor='white', margin=dict(l=20,r=20,t=50,b=20), hovermode='closest'
        )
        fig.update_yaxes(scaleanchor='x', scaleratio=1)
        return fig

    # Group nodes by category for colored traces (NO labels; hover only)
    cat_vals = {}
    for n in tree.nodes:
        row = meta.get(str(n), {})
        val = row.get(attr)
        if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == '':
            v = 'Unspecified'
        else:
            v = str(val)
        cat_vals.setdefault(v, []).append(n)

    # Viridis mapping with semantic grouping
    cmap = _viridis_category_colors(list(cat_vals.keys()), attr)

    traces = [edge_trace]  # edges behind
    for cat in sorted(cat_vals.keys(), key=lambda x: (x=='Unspecified', x)):
        nodes = cat_vals[cat]
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        hovers = [hover_for(n) for n in nodes]
        sizes = [size_map.get(n, 10.0) for n in nodes]
        traces.append(
            go.Scatter(
                x=xs, y=ys, mode='markers',
                marker=dict(size=sizes, color=cmap.get(cat, '#1f77b4')),
                name=f"{attr.capitalize()}: {cat}", hoverinfo='text', text=hovers,
                customdata=nodes,
            )
        )

    title = f"Radial Organization Chart (colored by {attr})"
    fig = go.Figure(data=traces)
    fig.update_layout(
        showlegend=True,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor='white', margin=dict(l=20,r=20,t=50,b=20), title=title,
        hovermode='closest'
    )
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    return fig


# -------------------------
# Filtering helpers (category-driven relayout with rewiring)
# -------------------------

def _normalize_category(val):
    if pd.isna(val) or str(val).strip() == '':
        return 'Unspecified'
    return str(val)


def rewire_df_by_categories(df: pd.DataFrame, attr: str, keep_categories: list) -> pd.DataFrame:
    """Filter rows by category and **rewire** nodes to nearest kept ancestor.
    - Always keep the CEO.
    - Keep employees whose own category is selected.
    - If a node's manager is filtered out, attach the node to the nearest kept ancestor up the chain.
    """
    if attr not in df.columns:
        return df.copy()

    # Clean trailing whitespace first
    df2 = clean_trailing_whitespace(df.copy())

    def _to_null_or_str(x):
        if pd.isna(x) or x is None:
            return None
        s = str(x).strip()
        if s == '' or s.lower() in {'nan', 'none'}:
            return None
        return s

    def _to_str(x):
        return str(x).strip()

    df2['employee'] = df2['employee'].apply(_to_str)
    df2['manager'] = df2['manager'].apply(_to_null_or_str)

    # Identify CEO
    ceo_rows = df2[df2['manager'].isna()]
    if ceo_rows.empty:
        return df.copy()
    ceo = ceo_rows.iloc[0]['employee']

    # Normalize categories (based on original df values, cleaned)
    df2['_cat'] = df2[attr].apply(_normalize_category) if attr in df2.columns else 'Unspecified'

    # Keep set = CEO + selected cats
    keep = set(df2.loc[(df2['_cat'].isin(keep_categories)) | (df2['employee'] == ceo), 'employee'])

    # Map for climbing ancestry
    manager_map = dict(zip(df2['employee'], df2['manager']))

    def nearest_kept_manager(e: str):
        m = manager_map.get(e)
        visited = set()
        while m is not None and m not in keep:
            if m in visited:  # safety against cycles
                return None
            visited.add(m)
            m = manager_map.get(m)
        return m  # may be None (becomes CEO if e==CEO)

    rows = []
    for _, row in df2[df2['employee'].isin(keep)].iterrows():
        e = row['employee']
        new_m = None if e == ceo else nearest_kept_manager(e)
        rd = {k: row[k] for k in df2.columns if k in df2}  # preserve cols
        rd['employee'] = e
        rd['manager'] = new_m
        rows.append(rd)

    kept_df = pd.DataFrame(rows)
    kept_df.loc[kept_df['employee'] == ceo, 'manager'] = None
    kept_df = kept_df.drop_duplicates(subset=['employee'], keep='first').reset_index(drop=True)
    return kept_df


# -------------------------
# Dash app (reads local sample if nothing uploaded)
# -------------------------
app: Dash = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Radial Org Chart (Rule-based Layout)"),
    html.P([
        "Upload a CSV with columns: employee, manager (CEO has blank manager). ",
        "Optional columns: ", html.Code("role"), " and ", html.Code("team"), ". ",
        "If you don't upload a file, the app will attempt to read ", html.Code("sample_250.csv"), " from the current working directory."
    ]),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
        style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
        multiple=False
    ),
    html.Div([
        html.Label("Color by:"),
        dcc.Dropdown(id='color-by', options=[
            {"label": "None", "value": "none"},
            {"label": "Role", "value": "role"},
            {"label": "Team", "value": "team"},
        ], value='none', clearable=False, style={"maxWidth": "240px"}),
    ], style={"margin": "8px 0"}),

    html.Div(id='category-filter-container', children=[
        html.Label("Categories:"),
        dcc.Dropdown(id='category-filter', options=[], value=[], multi=True, clearable=False, placeholder="Select categories…", style={"maxWidth": "520px"}),
    ], style={"display": "none", "margin": "4px 0 12px"}),

    # --- Node size controls ---
    html.Div([
        html.Label("Node size range"),
        html.Div([
            html.Div([
                html.Div("Max size"),
                dcc.Slider(id='max-size', min=8, max=40, step=1, value=22, updatemode='drag',
                           marks={8:'8', 16:'16', 22:'22', 30:'30', 40:'40'}, tooltip={"placement":"bottom"}),
            ], style={"flex": 1, "marginRight": "12px"}),
            html.Div([
                html.Div("Min size"),
                dcc.Slider(id='min-size', min=3, max=25, step=1, value=7, updatemode='drag',
                           marks={3:'3', 7:'7', 12:'12', 18:'18', 25:'25'}, tooltip={"placement":"bottom"}),
            ], style={"flex": 1}),
        ], style={"display": "flex", "gap": "12px"}),
    ], style={"margin": "6px 0 10px"}),

    # --- Graph + Stats side-by-side ---
    html.Div([
        html.Div([
            dcc.Graph(id='org-graph', style={'height': '80vh'}),
        ], style={"flex": 3}),
        html.Div([
            html.H4("Statistics"),
            html.Div(id='overall-stats', style={"marginBottom": "8px"}),
            html.Hr(),
            html.Div(id='node-stats'),
        ], style={"flex": 1, "minWidth": "280px", "borderLeft": "1px solid #eee", "padding": "8px"}),
    ], style={"display": "flex", "gap": "12px"}),

    html.Details([
        html.Summary("Filtered data (debug)"),
        html.Pre(id='debug-df', style={"whiteSpace": "pre-wrap", "fontFamily": "monospace", "background": "#fafafa", "padding": "8px", "border": "1px solid #eee"}),
    ], open=False, style={"marginTop": "10px"}),

    dcc.Store(id='csv-contents'),
    dcc.Store(id='plot-df-store'),  # filtered/rewired DataFrame for stats callbacks
])


@app.callback(Output('csv-contents', 'data'), Input('upload-data', 'contents'))
def keep_csv(contents):
    return contents


# Populate the category filter based on the data and the selected color_by
@app.callback(
    Output('category-filter-container', 'style'),
    Output('category-filter', 'options'),
    Output('category-filter', 'value'),
    Input('csv-contents', 'data'),
    Input('color-by', 'value'),
)
def populate_category_filter(contents, color_by):
    # Default hidden state
    hidden_style = {"display": "none", "margin": "4px 0 12px"}
    shown_style = {"display": "block", "margin": "4px 0 12px"}

    if color_by in (None, 'none'):
        return hidden_style, [], []

    # Load df
    try:
        if contents is None:
            df = pd.read_csv('sample_250.csv')
        else:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception:
        return hidden_style, [], []

    # Clean trailing whitespace globally
    df = clean_trailing_whitespace(df)

    if color_by not in df.columns:
        return hidden_style, [], []

    cats = df[color_by].apply(_normalize_category).unique().tolist()
    cats = sorted([c for c in cats if c != 'Unspecified']) + (['Unspecified'] if 'Unspecified' in cats else [])
    options = [{"label": c, "value": c} for c in cats]
    return shown_style, options, cats  # default: all selected


# Main figure update with filtering, rewiring, fixed rings, and debug printing
@app.callback(
    Output('org-graph', 'figure'),
    Output('debug-df', 'children'),
    Output('plot-df-store', 'data'),
    Input('csv-contents', 'data'),
    Input('color-by', 'value'),
    Input('category-filter', 'value'),
    Input('min-size', 'value'),
    Input('max-size', 'value'),
)
def update_figure(contents, color_by, selected_categories, min_size_val, max_size_val):
    def df_to_debug_text(df_: pd.DataFrame, note: str = "") -> str:
        cols = [c for c in ['employee','manager','role','team'] if c in df_.columns]
        head = df_[cols] if cols else df_
        text = head.to_csv(index=False)
        meta = f"Rows: {len(df_)}, Cols: {list(df_.columns)}"
        return (note + "\n" + meta + "\n\n" + text).strip()

    # sanitize size range (ensure min <= max)
    try:
        min_s = float(min_size_val) if min_size_val is not None else 7.0
        max_s = float(max_size_val) if max_size_val is not None else 22.0
    except Exception:
        min_s, max_s = 7.0, 22.0
    if min_s > max_s:
        min_s, max_s = max_s, min_s
    # clamp to reasonable bounds
    min_s = max(1.0, min(60.0, min_s))
    max_s = max(1.0, min(60.0, max_s))

    # read data (uploaded or local sample.csv)
    if contents is None:
        try:
            df_full = pd.read_csv('sample_250.csv')
        except Exception as e:
            fig = go.Figure()
            msg = f"No upload and failed to read 'sample.csv': {e}"
            fig.update_layout(title=msg)
            return fig, msg, None
    else:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df_full = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            fig = go.Figure()
            msg = f"Error reading uploaded CSV: {e}"
            fig.update_layout(title=msg)
            return fig, msg, None

    # Clean trailing whitespace globally on the full dataset
    df_full = clean_trailing_whitespace(df_full)

    # Build the FULL tree once to get fixed original rings
    try:
        full_tree = build_tree_from_df(df_full.copy())
        depth_lookup = full_tree.depths()  # original depths by employee
    except Exception as e:
        fig = go.Figure()
        msg = f"Data error in full dataset: {e}\n\n" + df_to_debug_text(df_full, note="Full dataset")
        fig.update_layout(title=f"Data error: {e}")
        print(msg)
        return fig, msg, None

    # Normalize color_by
    cb = None if not color_by or color_by == 'none' else color_by

    # If coloring and user selected categories, REWIRE DF and relayout
    df_for_plot = df_full.copy()
    debug_note = "Using full dataset (no category filtering)"
    if cb and isinstance(selected_categories, list) and len(selected_categories) > 0:
        df_for_plot = rewire_df_by_categories(df_full, cb, selected_categories)
        debug_note = f"Rewired by {cb}: {selected_categories}"
        if df_for_plot.empty:
            fig = go.Figure()
            msg = df_to_debug_text(df_for_plot, note=debug_note + " — result is EMPTY")
            fig.update_layout(title="No nodes to display after filtering.")
            print(msg)  # server log
            return fig, msg, []

    try:
        tree = build_tree_from_df(df_for_plot.copy())
    except Exception as e:
        fig = go.Figure()
        msg = f"Data error (filtered/rewired): {e}\n\n" + df_to_debug_text(df_for_plot, note=debug_note)
        fig.update_layout(title=f"Data error: {e}")
        print(msg)  # server log
        return fig, msg, []

    pos = radial_layout_rules(tree, radius_step=1.0, depth_lookup=depth_lookup)

    fig = org_to_figure(tree, pos, df_for_plot, cb, min_size=min_s, max_size=max_s)

    debug_text = df_to_debug_text(df_for_plot, note=debug_note + f"\nNode size range: {min_s}–{max_s}")
    print(debug_text)  # also log to console for debugging

    # Store filtered DF as records for downstream stats callbacks
    records = df_for_plot.to_dict(orient='records')
    return fig, debug_text, records


# -------------------------
# Stats panel callbacks
# -------------------------

def _depth_counts(tree: OrgTree) -> dict[int, int]:
    d = tree.depths()
    counts = defaultdict(int)
    for depth in d.values():
        counts[depth] += 1
    return dict(sorted(counts.items()))


@app.callback(
    Output('overall-stats', 'children'),
    Output('node-stats', 'children'),
    Input('plot-df-store', 'data'),
    Input('org-graph', 'hoverData'),
)
def update_stats(records, hoverData):
    # Overall stats (by current displayed graph)
    if not records:
        return html.Div("Load data to see statistics."), html.Div("Hover a node to see details.")

    df = pd.DataFrame(records)
    try:
        tree = build_tree_from_df(df.copy())
    except Exception as e:
        return html.Div(f"Stats unavailable: {e}"), html.Div()

    counts = _depth_counts(tree)
    total = sum(counts.values())
    max_depth = max(counts.keys()) if counts else 0

    overall = html.Div([
        html.Div(f"Total employees (shown): {total}"),
        html.Div(f"Max depth: {max_depth}"),
        html.Table([
            html.Thead(html.Tr([html.Th("Depth"), html.Th("Count")])) ,
            html.Tbody([html.Tr([html.Td(d), html.Td(c)]) for d, c in counts.items()])
        ], style={"marginTop": "6px", "borderCollapse": "collapse", "width": "100%"})
    ])

    # Hovered node stats
    if not hoverData or 'points' not in hoverData or not hoverData['points']:
        node_info = html.Div("Hover a node to see details.")
        return overall, node_info

    # Pull the employee name from customdata
    point = hoverData['points'][0]
    emp = point.get('customdata')
    if emp is None:
        node_info = html.Div("Hover a node to see details.")
        return overall, node_info

    # Compute direct children and total descendants
    sizes = _subtree_sizes(tree)
    depths_map = tree.depths()
    direct = len(tree.children.get(emp, []))
    descendants = max(0, sizes.get(emp, 1) - 1)
    depth_val = depths_map.get(emp, None)

    node_info = html.Div([
        html.Strong(emp),
        html.Ul([
            html.Li(f"Depth: {depth_val}"),
            html.Li(f"Direct reports: {direct}"),
            html.Li(f"Total descendants: {descendants}"),
        ])
    ])

    return overall, node_info


if __name__ == '__main__':
    app.run(debug=True)
