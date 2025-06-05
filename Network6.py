import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
import tempfile

st.set_page_config(page_title="Theme Network Explorer", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Themes_Final.csv")
    for col in ['text', 'summary', 'large_theme', 'macro_theme_name']:
        df[col] = df[col].astype(str).fillna('')
    return df

@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def make_theme_network(theme_names, theme_texts, theme_counts, threshold=0.2):
    net = Network(
        height='700px',
        width='100%',
        bgcolor='#ffffff',
        font_color='black'
    )
    model = get_model()
    embeddings = model.encode([theme_texts[name] for name in theme_names])
    sim_matrix = cosine_similarity(embeddings)
    min_count = min(theme_counts.values())
    max_count = max(theme_counts.values())
    for name in theme_names:
        prompt_count = theme_counts[name]
        node_size = int(np.interp(np.log1p(prompt_count),
                                  [np.log1p(min_count), np.log1p(max_count)],
                                  [45, 170]))
        net.add_node(
            name,
            label=name,
            size=node_size,
            title=f"{name.upper()}\nNumber of prompts: {prompt_count}",
            color={'background': '#97C2FC', 'border': '#2B7CE9'},
            font={'size': 28, 'color': 'black', 'face': 'Arial', 'bold': True},
            borderWidth=2,
            shadow=True
        )
    for i in range(len(theme_names)):
        for j in range(i+1, len(theme_names)):
            if sim_matrix[i, j] > threshold:
                similarity_score = float(sim_matrix[i, j])
                if similarity_score > 0.7:
                    edge_color = '#FF6B6B'
                    width = 4
                elif similarity_score > 0.5:
                    edge_color = '#4ECDC4'
                    width = 3
                else:
                    edge_color = '#95A5A6'
                    width = 2
                net.add_edge(
                    theme_names[i],
                    theme_names[j],
                    value=similarity_score,
                    title=f"Similarity: {similarity_score:.3f}",
                    color={'color': edge_color},
                    width=width,
                    smooth={'type': 'continuous'}
                )
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -200,
          "centralGravity": 0.002,
          "springLength": 400,
          "springConstant": 0.02,
          "avoidOverlap": 1
        },
        "minVelocity": 0.7,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 300}
      },
      "edges": {
        "smooth": true,
        "font": {"size": 10, "color": "black"}
      },
      "nodes": {
        "font": {"size": 28, "color": "black", "face": "Arial", "bold": true},
        "borderWidth": 2,
        "shadow": true
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": false
      }
    }
    """)
    return net

def display_network(net):
    path = tempfile.mktemp(suffix=".html")
    net.save_graph(path)
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=700, scrolling=False)

def main():
    df = load_data()
    st.sidebar.header("Network Information")
    st.sidebar.markdown("**Fixed Settings:**")
    st.sidebar.markdown("Similarity Threshold: **0.2**")
    st.sidebar.markdown("**Legend:**")
    st.sidebar.markdown("Strong similarity (>0.7): Red")
    st.sidebar.markdown("Medium similarity (0.5-0.7): Teal")
    st.sidebar.markdown("Weak similarity (<0.5): Gray")
    st.sidebar.markdown("**Node size = Number of prompts**")

    if 'level' not in st.session_state:
        st.session_state.level = "macro"
    if 'selected_macro' not in st.session_state:
        st.session_state.selected_macro = None
    if 'selected_micro' not in st.session_state:
        st.session_state.selected_micro = None

    if st.session_state.level == "macro":
        st.title("Macro Themes")
        st.write("Node size represents the number of prompts in each theme. Larger nodes = more prompts.")
        macro_theme_counts = df['large_theme'].value_counts().to_dict()
        macro_theme_names_sorted = sorted(macro_theme_counts, key=lambda x: macro_theme_counts[x], reverse=True)
        macro_theme_texts = df.groupby('large_theme')['text'].apply(lambda x: ' '.join(x)).to_dict()
        max_prompts = max(macro_theme_counts.values())
        min_prompts = min(macro_theme_counts.values())
        st.info(f"Theme sizes range from {min_prompts} to {max_prompts} prompts")
        net = make_theme_network(macro_theme_names_sorted, macro_theme_texts, macro_theme_counts, 0.2)
        display_network(net)

        st.sidebar.markdown(
            '<div style="margin-bottom:0px;"><b style="font-size:20px;">Select a Macro Theme</b></div>',
            unsafe_allow_html=True
        )
        selected = st.sidebar.selectbox(
            "",
            macro_theme_names_sorted,
            help="Select a macro theme to see its micro themes"
        )
        if st.sidebar.button("Explore Micro Themes"):
            st.session_state.level = "micro"
            st.session_state.selected_macro = selected
            st.rerun()

    elif st.session_state.level == "micro":
        selected_macro = st.session_state.selected_macro
        st.title(f"Micro Themes in '{selected_macro}'")
        st.write("Node size represents the number of prompts in each micro theme.")
        micro_data = df[df['large_theme'] == selected_macro]
        micro_theme_counts = micro_data['macro_theme_name'].value_counts().to_dict()
        micro_theme_names_sorted = sorted(micro_theme_counts, key=lambda x: micro_theme_counts[x], reverse=True)
        micro_theme_texts = micro_data.groupby('macro_theme_name')['text'].apply(lambda x: ' '.join(x)).to_dict()
        if not micro_theme_names_sorted:
            st.warning(f"No micro themes found for '{selected_macro}'")
            if st.sidebar.button("Back to Macro Themes"):
                st.session_state.level = "macro"
                st.rerun()
            return
        max_prompts = max(micro_theme_counts.values())
        min_prompts = min(micro_theme_counts.values())
        st.info(f"Micro theme sizes range from {min_prompts} to {max_prompts} prompts")
        net = make_theme_network(micro_theme_names_sorted, micro_theme_texts, micro_theme_counts, 0.2)
        display_network(net)

        st.sidebar.markdown(
            '<div style="margin-bottom:0px;"><b style="font-size:20px;">Select a Micro Theme</b></div>',
            unsafe_allow_html=True
        )
        selected = st.sidebar.selectbox(
            "",
            micro_theme_names_sorted,
            help="Select a micro theme to see all prompts within it"
        )
        if st.sidebar.button("View Prompts"):
            st.session_state.level = "prompts"
            st.session_state.selected_micro = selected
            st.rerun()
        if st.sidebar.button("Back to Macro Themes"):
            st.session_state.level = "macro"
            st.rerun()

    elif st.session_state.level == "prompts":
        selected_macro = st.session_state.selected_macro
        selected_micro = st.session_state.selected_micro
        st.title(f"Prompts in '{selected_micro}'")
        st.write("All prompts in this micro theme (scroll to view):")
        prompts_data = df[
            (df['large_theme'] == selected_macro) &
            (df['macro_theme_name'] == selected_micro)
        ]
        prompts = prompts_data['text'].tolist()
        st.write(f"Total prompts: {len(prompts)}")
        for i, prompt in enumerate(prompts, 1):
            st.markdown(f"""
            <div style="padding:12px; margin-bottom:8px; background:#f8f9fa; border-radius:5px; border:1px solid #e3e3e3;">
            <b>Prompt {i}:</b><br>{prompt}
            </div>
            """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Back to Micro Themes"):
                st.session_state.level = "micro"
                st.rerun()
        with col2:
            if st.button("Back to Macro Themes"):
                st.session_state.level = "macro"
                st.rerun()

if __name__ == "__main__":
    main()
