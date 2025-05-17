
import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title='Hodge Mirror Explorer', layout='wide')

examples = json.loads(Path('complex_manifolds.json').read_text())

if 'custom_examples' not in st.session_state:
    st.session_state['custom_examples'] = []
examples.extend(st.session_state['custom_examples'])

def calculate_griffiths_residue(hodge, dim):
    try:
        return hodge[dim-1][1] - hodge[dim-2][2]
    except:
        return "N/A"

def is_mirror_pair(h1, h2, dim):
    return all(
        h1[p][q] == h2[dim-p][q]
        for p in range(dim+1) for q in range(dim+1)
        if p + q <= dim
    )

def find_mirror_partner(example, examples):
    dim = example['dimension']
    for ex in examples:
        if ex['id'] != example['id'] and is_mirror_pair(example['hodgeNumbers'], ex['hodgeNumbers'], dim):
            return ex
    return None

def render_hodge_diamond(hodge, dim, dark):
    for p in range(dim+1):
        row = []
        for q in range(dim+1):
            if p + q <= dim:
                row.append(f"<div class='cell'>{hodge[p][q]}</div>")
        if row:
            st.markdown(f"<div class='row'>{''.join(row)}</div>", unsafe_allow_html=True)

def render_betti_numbers(betti, dark, label=""):
    bg = '#1f2937' if dark else '#f0f4ff'
    text = '#e5e7eb' if dark else '#111827'
    st.markdown(f"### Betti Numbers {label}")
    st.markdown(
        "<div style='display:flex; gap:12px; flex-wrap:wrap;'>" +
        "".join([
            f"<div style='text-align:center; background:{bg}; color:{text}; padding:10px; border-radius:8px'>             <div>\(b_{{{i}}}\)</div><div style='font-weight:bold'>{b}</div></div>"
            for i, b in enumerate(betti)
        ]) +
        "</div>",
        unsafe_allow_html=True
    )


import matplotlib.pyplot as plt
import seaborn as sns

def plot_hodge_heatmap(hodge, title, dark):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = 'Blues' if not dark else 'viridis'
    sns.heatmap(hodge, annot=True, fmt='d', cbar=True, square=True,
                linewidths=.5, cmap=cmap, ax=ax)
    ax.set_title(title, fontsize=14)
    st.pyplot(fig)

def plot_betti_comparison(betti1, betti2, dark):
    fig, ax = plt.subplots()
    indices = list(range(len(betti1)))
    ax.plot(indices, betti1, label='Original', marker='o')
    ax.plot(indices, betti2, label='Mirror', marker='x')
    ax.set_xlabel('k')
    ax.set_ylabel('bₖ')
    ax.set_title('Betti Number Comparison')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)



import pandas as pd
from io import BytesIO

def get_mirror_pair_table(examples):
    pairs = []
    seen = set()
    for ex in examples:
        mirror = find_mirror_partner(ex, examples)
        if mirror and frozenset((ex['id'], mirror['id'])) not in seen:
            seen.add(frozenset((ex['id'], mirror['id'])))
            pairs.append({
                'Manifold A': ex['name'],
                'Manifold B': mirror['name'],
                'Dimension': ex['dimension']
            })
    return pd.DataFrame(pairs)

def plot_dimension_histogram(examples):
    dims = [ex['dimension'] for ex in examples]
    fig, ax = plt.subplots()
    sns.histplot(dims, bins=range(min(dims), max(dims)+2), kde=False, ax=ax)
    ax.set_title("Distribution of Manifold Dimensions")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def export_plot_as_pdf(fig, title="export"):
    buf = BytesIO()
    fig.savefig(buf, format="pdf")
    st.download_button("📄 Download as PDF", data=buf.getvalue(), file_name=f"{title}.pdf", mime="application/pdf")



from sklearn.decomposition import PCA
import plotly.express as px

def export_examples_csv(examples):
    export_custom_manifolds()
    rows = []
    for ex in examples:
        rows.append({
            "ID": ex["id"],
            "Name": ex["name"],
            "Dimension": ex["dimension"],
            "Betti Numbers": ex["bettiNumbers"],
            "Properties": "; ".join(ex["properties"])
        })
    df = pd.DataFrame(rows)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📁 Download Full Example Data (CSV)", csv, "complex_manifolds.csv", "text/csv")

def plot_3d_embedding(examples):
    betti_matrix = [ex["bettiNumbers"][:13] for ex in examples]
    pca = PCA(n_components=3)
    coords = pca.fit_transform(betti_matrix)
    names = [ex["name"] for ex in examples]
    fig = px.scatter_3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], text=names,
                        color=[ex["dimension"] for ex in examples],
                        labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                        title="3D PCA of Betti Numbers")
    st.plotly_chart(fig, use_container_width=True)



import numpy as np

def render_custom_manifold_input():
    with st.expander("➕ Add Custom Complex Manifold"):
        name = st.text_input("Name", placeholder="e.g., My Custom Manifold")
        dimension = st.number_input("Dimension", min_value=1, max_value=20, value=3)
        betti_input = st.text_input("Betti Numbers (comma-separated)", placeholder="e.g., 1,0,2,0,1")
        description = st.text_area("Description", placeholder="Describe the manifold here...")
        props = st.text_area("Properties (semicolon-separated)", placeholder="e.g., Kähler; Simply connected")

        hodge_matrix = []
        st.markdown("### Enter Hodge Diamond Values (Upper Triangle Only)")
        for p in range(int(dimension)+1):
            cols = st.columns(p + 1)
            row = []
            for q in range(int(dimension)+1):
                if p + q <= dimension:
                    val = cols[q].number_input(f"h^{p},{q}", min_value=0, max_value=999999, value=0, key=f"h-{p}-{q}")
                    row.append(val)
                else:
                    row.append(0)
            hodge_matrix.append(row)

        if st.button("Add to Session"):
            try:
                betti_list = list(map(int, betti_input.split(",")))
                new_manifold = {
                    "id": f"custom-{name.lower().replace(' ', '-')}",
                    "name": name,
                    "dimension": int(dimension),
                    "bettiNumbers": betti_list,
                    "hodgeNumbers": hodge_matrix,
                    "description": description,
                    "properties": props.split(";")
                }
                st.session_state["custom_examples"].append(new_manifold)
                st.success(f"Added custom manifold: {name}")
            except:
                st.error("Invalid input! Check that Betti numbers are integers.")



def export_custom_manifolds():
    if st.session_state['custom_examples']:
        custom_json = json.dumps(st.session_state['custom_examples'], indent=2)
        st.download_button("💾 Download Custom Manifolds (JSON)", custom_json, "custom_manifolds.json", "application/json")

def render_hodge_diamond_with_tooltip(hodge, dim, dark):
    for p in range(dim+1):
        row = []
        for q in range(dim+1):
            if p + q <= dim:
                tooltip = f"\\( h^{{{p},{q}}} \\)"
                row.append(f"<div class='cell' title='{tooltip}'>{hodge[p][q]}</div>")
        if row:
            st.markdown(f"<div class='row'>{''.join(row)}</div>", unsafe_allow_html=True)



def import_custom_manifolds():
    uploaded_file = st.file_uploader("📂 Upload Custom Manifolds JSON", type="json", key="custom_json_upload")
    if uploaded_file:
        try:
            imported = json.load(uploaded_file)
            st.session_state['custom_examples'].extend(imported)
            st.success(f"Imported {len(imported)} custom manifolds.")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

def auto_load_custom_manifolds():
    local_path = Path("custom_manifolds.json")
    if local_path.exists() and 'auto_loaded' not in st.session_state:
        try:
            loaded = json.loads(local_path.read_text())
            st.session_state['custom_examples'].extend(loaded)
            st.session_state['auto_loaded'] = True
            st.toast(f"Auto-loaded {len(loaded)} custom manifolds from file.", icon="📁")
        except:
            st.warning("Could not auto-load saved custom manifolds.")


auto_load_custom_manifolds()
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

# CSS block
st.markdown(f'''
    <style>
    .cell {{
      display: inline-block;
      width: 50px;
      height: 50px;
      margin: 2px;
      text-align: center;
      line-height: 50px;
      font-weight: bold;
      border-radius: 8px;
      background: {'#1f2937' if dark_mode else '#f0f4ff'};
      color: {'#f9fafb' if dark_mode else '#1e293b'};
    }}
    .row {{
      text-align: center;
    }}
    </style>
''', unsafe_allow_html=True)

st.title("🔁 Hodge Mirror Symmetry Explorer")


tab1, tab2, tab3 = st.tabs(["🔍 Explore", "📊 Statistics", "📦 Dataset"])

with tab1:
    names = [ex['name'] for ex in examples]
    selected_name = st.selectbox("Choose a complex manifold", names, key="selectbox_tab1")
    selected = next(ex for ex in examples if ex['name'] == selected_name)

    max_dim = selected['dimension']
    selected_dim = st.slider("Select dimension to view", 1, max_dim, value=max_dim, key="slider_tab1")

    col1, col2 = st.columns([2, 3])

    with col1:
        st.header("📘 Manifold Info")
        st.subheader(selected['name'])
        st.markdown(f"**Full Dimension:** {selected['dimension']}")
        st.markdown(f"**Viewing Dimension:** {selected_dim}")
        st.markdown(f"{selected['description']}")
        st.markdown("### Properties:")
        st.markdown("\n".join([f"- {p}" for p in selected['properties']]))
        render_betti_numbers(selected['bettiNumbers'], dark_mode)

    with col2:
        st.header("🔷 Hodge Diamond")
        render_hodge_diamond_with_tooltip(selected['hodgeNumbers'], selected_dim, dark_mode)

    st.divider()
    st.subheader("🧪 Griffiths Residue")
    res = calculate_griffiths_residue(selected['hodgeNumbers'], selected_dim)
    st.info(f"h^({selected_dim-1},1) - h^({selected_dim-2},2) = {res}")

    st.divider()
    mirror = find_mirror_partner(selected, examples)
    if mirror:
        st.subheader("🔁 Mirror Symmetry")
        st.markdown(f"**Mirror Partner:** *{mirror['name']}*")
        colm1, colm2 = st.columns(2)
        with colm1:
            st.markdown("#### Original Hodge Diamond")
            render_hodge_diamond_with_tooltip(selected['hodgeNumbers'], selected_dim, dark_mode)
            render_betti_numbers(selected['bettiNumbers'], dark_mode, label="(Original)")
            plot_hodge_heatmap(selected['hodgeNumbers'], f"{selected['name']} - Hodge Heatmap", dark_mode)

        with colm2:
            st.markdown("#### Mirror Hodge Diamond")
            render_hodge_diamond_with_tooltip(mirror['hodgeNumbers'], selected_dim, dark_mode)
            render_betti_numbers(mirror['bettiNumbers'], dark_mode, label="(Mirror)")
            plot_hodge_heatmap(mirror['hodgeNumbers'], f"{mirror['name']} - Mirror Heatmap", dark_mode)

        st.subheader("📊 Betti Number Comparison")
        plot_betti_comparison(selected['bettiNumbers'], mirror['bettiNumbers'], dark_mode)
    else:
        st.info("No mirror partner found in dataset.")

with tab2:
    st.subheader("📊 Mirror Pair Summary Table")
    mirror_df = get_mirror_pair_table(examples)
    st.dataframe(mirror_df)

    st.subheader("📈 Dimension Histogram of All Manifolds")
    plot_dimension_histogram(examples)

    st.subheader("🧬 Betti Number Embedding (3D PCA)")
    plot_3d_embedding(examples)

with tab3:
    st.subheader("📥 Export Dataset")
    export_examples_csv(examples)
    export_custom_manifolds()

selected_name = st.selectbox("Choose a complex manifold", names, key="selectbox_bottom")
selected = next(ex for ex in examples if ex['name'] == selected_name)

max_dim = selected['dimension']
selected_dim = st.slider("Select dimension to view", 1, max_dim, value=max_dim, key="slider_bottom")

col1, col2 = st.columns([2, 3])

with col1:
    st.header("📘 Manifold Info")
    st.subheader(selected['name'])
    st.markdown(f"**Full Dimension:** {selected['dimension']}")
    st.markdown(f"**Viewing Dimension:** {selected_dim}")
    st.markdown(f"{selected['description']}")
    st.markdown("### Properties:")
    st.markdown("\n".join([f"- {p}" for p in selected['properties']]))
    render_betti_numbers(selected['bettiNumbers'], dark_mode)

with col2:
    st.header("🔷 Hodge Diamond")
    render_hodge_diamond_with_tooltip(selected['hodgeNumbers'], selected_dim, dark_mode)

st.divider()
st.subheader("🧪 Griffiths Residue")
res = calculate_griffiths_residue(selected['hodgeNumbers'], selected_dim)
st.info(f"h^({selected_dim-1},1) - h^({selected_dim-2},2) = {res}")

st.divider()
mirror = find_mirror_partner(selected, examples)
if mirror:
    st.subheader("🔁 Mirror Symmetry")
    st.markdown(f"**Mirror Partner:** *{mirror['name']}*")
    colm1, colm2 = st.columns(2)
    with colm1:
        st.markdown("#### Original Hodge Diamond")
        render_hodge_diamond_with_tooltip(selected['hodgeNumbers'], selected_dim, dark_mode)
        render_betti_numbers(selected['bettiNumbers'], dark_mode, label="(Original)")

    with colm1:
        st.markdown("#### Original Hodge Diamond")
        render_hodge_diamond_with_tooltip(selected['hodgeNumbers'], selected_dim, dark_mode)
        render_betti_numbers(selected['bettiNumbers'], dark_mode, label="(Original)")
        plot_hodge_heatmap(selected['hodgeNumbers'], f"{selected['name']} - Hodge Heatmap", dark_mode)

    with colm2:
        st.markdown("#### Mirror Hodge Diamond")
        plot_hodge_heatmap(mirror['hodgeNumbers'], f"{mirror['name']} - Mirror Heatmap", dark_mode)

    st.subheader("📊 Betti Number Comparison")
    plot_betti_comparison(selected['bettiNumbers'], mirror['bettiNumbers'], dark_mode)

else:
    st.info("No mirror partner found in dataset.")

st.divider()
st.subheader("📊 Mirror Pair Summary Table")
mirror_df = get_mirror_pair_table(examples)
st.dataframe(mirror_df)

st.subheader("📈 Dimension Histogram of All Manifolds")
plot_dimension_histogram(examples)
