import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from math import ceil
from itertools import islice
import plotly.express as px

# Helper Functions.
def open_image(x):
    try:
        return Image.open(os.path.join(data_path, x))
    except:
        return None

# Configure data path and cluster output path.
data_directory_path = 'data/'
cluster_output_path = 'clustering_output/'
cluster_stats_path = 'clustering_stats/'

# Streamlit configs.
logo_url = 'ui/logo-no-background.png'
st.set_page_config(page_title='VisualInsights', page_icon=logo_url, layout="wide", initial_sidebar_state="auto", menu_items=None)
st.header('Team 165 - CSE6242')
st.sidebar.image(logo_url)
exploration_tab, stats_tab = st.tabs(["Exploration", "Statistics"])

# Set image width and height.
fixed_width = 50
fixed_height = 50

# List of available datasets under data_directory path for user to select.
dataset_names_dir = [folder for folder in os.listdir(data_directory_path) if os.path.isdir(os.path.join(data_directory_path, folder))]
dataset_selected = st.sidebar.selectbox(
    "Choose Dataset:",
    dataset_names_dir
)

data_path = data_directory_path + dataset_selected + '/'

with st.sidebar:
    cluster_alg = st.radio(
        'Choose Clustering Method:',
        ('Agglomerative', 'Birch', 'KMeans', 'KMeans_PCA')
    )

cluster_select_path =  cluster_output_path + dataset_selected + '/' + cluster_alg + '.csv'

lookup_table = pd.read_csv(cluster_select_path)
header_row = lookup_table.columns
header_row_numeric = pd.to_numeric(header_row, errors='coerce')

max_cluster = np.nanmax(header_row_numeric)
min_cluster = np.nanmin(header_row_numeric)

with st.sidebar:
    numClusters = st.slider('Select Number of Clusters:', min_value = int(min_cluster), max_value = int(max_cluster))

filtered_data = lookup_table.filter(['file_name', str(numClusters)])
clusters = np.unique(filtered_data[str(numClusters)].values)
count_per_cluster = filtered_data.groupby([str(numClusters)]).count()
count_per_cluster.columns = ['Cluster Count']

with st.sidebar:
    st.table(count_per_cluster)

with exploration_tab:
    col_row={}
    for i in range(0, numClusters, 2):
        x, y = st.columns(2)
        col_row[clusters[i]] = x
        try:
            col_row[clusters[i+1]] = y
        except:
            pass

    limited_file_names = filtered_data.groupby(str(numClusters))['file_name'].head(40).tolist()

    filtered_data['file_name_image'] = filtered_data['file_name'].apply(lambda x: open_image(x) if x in limited_file_names else None)
    filtered_data = filtered_data.dropna(subset=['file_name_image'])
    image_list = filtered_data.groupby([str(numClusters)])['file_name_image'].apply(list)
    for ind,val in col_row.items():
        with val:
            st.image(image_list[ind], width=fixed_width)

cluster_stats_select_path =  cluster_stats_path + dataset_selected + '/' + cluster_alg + '/' + str(numClusters) + '.csv'
cluster_stats_table = pd.read_csv(cluster_stats_select_path)

cluster_stats_table_class = cluster_stats_table[~cluster_stats_table.object.isin(['day', 'night', 'summer', 'winter'])]
cluster_stats_table_time_season = cluster_stats_table[cluster_stats_table.object.isin(['day', 'night', 'summer', 'winter'])]

cluster_stats_model_path =  cluster_stats_path + dataset_selected + '/trainval_class_counts.csv'
cluster_stats_model_perf = pd.read_csv(cluster_stats_model_path)

labels = cluster_stats_table_class.columns[2:].values.tolist()
types = cluster_stats_table_class.object.unique()
counts = cluster_stats_table_class.iloc[:,2:].fillna(0).values.T.tolist()
fig_bar = px.bar(
    pd.DataFrame(counts, columns=types, index=labels).reset_index().melt(id_vars="index"),
    x = "index",
    y = "value",
    color = "variable",
    color_discrete_sequence=px.colors.qualitative.Alphabet_r,
    labels = {
        "index": "Cluster",
        "value": "Count",
        "variable": "Classes"},
    title = "Class Distribution per Cluster"
)

fig_bar.update_xaxes(type='category')

group_color = {'day': '#FBE104', 'night': '#C81F1F', 'summer': '#DAF7A6' , 'winter': '#581845'}
labels = cluster_stats_table_time_season.columns[2:].values.tolist()
types = cluster_stats_table_time_season.object.unique()
counts = cluster_stats_table_time_season.iloc[:,2:].fillna(0).values.T.tolist()
fig_bar_2 = px.bar(
    pd.DataFrame(counts, columns=types, index=labels).reset_index().melt(id_vars="index"),
    x = "index",
    y = "value",
    color = "variable",
    color_discrete_map=group_color,
    barmode='group',
    labels = {
        "index": "Cluster",
        "value": "Count",
        "variable": "Time/Season"},
    title = "Time/Season Distribution per Cluster"
)

fig_model_bar = px.bar(
    cluster_stats_model_perf,
    y = "sub_category",
    x = "counts",
    hover_data=["proportion"],
    orientation = 'h',
    labels = {
        "sub_category": "Class Sub-Category",
        "counts": "Count"},
    title = "Counts/Proportion per Class Sub-Category"
)
fig_model_bar.add_vline(x=1500, line_width=2, opacity=0.3, line_dash="dash", line_color="black", annotation_text="Target Proportion")

with stats_tab:
    st.plotly_chart(fig_bar, theme="streamlit", use_container_width=True)
    st.plotly_chart(fig_bar_2, theme="streamlit", use_container_width=True)
    st.plotly_chart(fig_model_bar, theme="streamlit", use_container_width=True)

def _max_width_():
    max_width_str = "max-width: 1900px;"
    st.markdown(
        f"""
    <style>
    .css-1njjmvq {{
        border: 2px solid black;
    }}
    .css-1r6slb0 {{
        border: 2px solid black;
    }}
    .css-1kyxreq {{
        border: 2px solid black;
        gap: 0rem;
        place-items: center;
        margin: auto;
        display: flex;
        justify-content: center;
        padding-top: {0.2}rem;
        padding-bottom: {0.2}rem;
    }}
    .stApp [data-testid="stDecoration"]{{
        display:none;
    }}
    <style>
    """,
        unsafe_allow_html=True,
    )

_max_width_()