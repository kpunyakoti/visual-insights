import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from math import ceil

logo_url = 'logo-no-background.png'

st.set_page_config(page_title='VisualInsights', page_icon=logo_url, layout="wide", initial_sidebar_state="auto", menu_items=None)
st.sidebar.image(logo_url)

data_directory_path = '../data/'
dataset_names_dir = [folder for folder in os.listdir(data_directory_path) if os.path.isdir(os.path.join(data_directory_path, folder))]

add_selectbox = st.sidebar.selectbox(
    "Choose Dataset:",
    dataset_names_dir
)

data_path = '../data/' + add_selectbox + '/'
file_data_path = data_path + 'cluster_info.csv'

# image width and height.
fixed_width = 30
fixed_height = 30

lookup_table = pd.read_csv(file_data_path)
header_row = lookup_table.columns
header_row_numeric = pd.to_numeric(header_row, errors='coerce')

max_cluster = np.nanmax(header_row_numeric)
min_cluster = np.nanmin(header_row_numeric)

with st.sidebar:
    numClusters = st.slider('Select a Number of Clusters:', min_value = int(min_cluster), max_value = int(max_cluster))
    st.write('Number of clusters chosen', numClusters)

filtered_data = lookup_table.filter(['file_name', str(numClusters)])
clusters = np.unique(filtered_data[str(numClusters)].values)

col_row={}
for i in range(0, numClusters, 2):
    x, y = st.columns(2)
    col_row[clusters[i]] = x
    try:
        col_row[clusters[i+1]] = y
    except:
        pass

# for i in range(0, numClusters, 3):
#     x, y, z = st.columns(3)
#     col_row[clusters[i]] = x
#     try:
#         col_row[clusters[i+1]] = y        
#     except:
#         try:
#             col_row[clusters[i+2]] = z
#         except: 
#             pass

from itertools import islice

limited_file_names = list(islice(filtered_data['file_name'].tolist(), 608))

def open_image(x):
    try:
        return Image.open(os.path.join(data_path, x))
    except:
        return None

filtered_data['file_name_image'] = filtered_data['file_name'].apply(lambda x: open_image(x) if x in limited_file_names else None)

# Filter out rows with None in 'file_name_image'
filtered_data = filtered_data.dropna(subset=['file_name_image'])



# filtered_data['file_name_image'] = filtered_data['file_name'].apply(lambda x: Image.open(os.path.join(data_path,x)))
# .thumbnail(fixed_width, fixed_height))
image_list = filtered_data.groupby([str(numClusters)])['file_name_image'].apply(list)
#limitied_image_list = image_list[:2]

# for image in image_list:
#     resized_image = ImageOps.fit(image, (fixed_width, fixed_height), Image.ANTIALIAS)
#     # image_list.append(resized_image)

# print(image_list)

# # Concatenate images horizontally
# concatenated_image = ImageOps.concat(image_list)

# For the memories #
# print(col_row)
# filtered_data['colrow'] = filtered_data[str(numClusters)]
# filtered_data['colrow'] = filtered_data['colrow'].replace(col_row)
# print(filtered_data)
# for index, row in filtered_data.iterrows():
#     image_path = row['file_name']
#     cluster = row[str(numClusters)]
#     image = Image.open(os.path.join(data_path,image_path))
#     for ind,val in col_row.items():
#         if cluster == ind:
#             with val:
#                 st.image(image, width=50)
# image_dict = {}
# for _, row in filtered_data.iterrows():
#     image_path = row['file_name']
#     cluster = row[str(numClusters)]
#     image = Image.open(os.path.join(data_path,image_path))
#     image_dict[cluster] = image_list

for ind,val in col_row.items():
    with val:
        st.image(image_list[ind], width=fixed_width)

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
    }}
    .stApp [data-testid="stDecoration"]{{
        display:none;
    }}
    <style>
    """,
        unsafe_allow_html=True,
    )

# div[data-testid="column"]:nth-of-type(1)
#     {{
#         border:1px solid red;
#     }} 
#     div[data-testid="column"]:nth-of-type(2)
#     {{
#         border:1px solid blue;
#         text-align: end;
#     }} 

_max_width_()