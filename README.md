# VisualInsights
**CS-6242**
<br/> **Team 165**: VisualInsights
<br/> **Authors**: Mohammad Azzam, Zain Bitar, Sangeeth Nambiar, Krishna Punyakoti

## DESCRIPTION
Deep Learning and its applications have made tremendous progress in the last decade across various fields like computer vision and natural language processing (Heaton, 2018). It is well established that deep learning is data hungry (Aggarwal, 2018) and the performance of these models rely on the availability of large and high-quality datasets. Researchers have argued that data should be treated as “first class citizens” alongside data management leading to what is today referred to as “Data centric AI” (Whang et al., 2023). One of the key challenge AI practitioners face today to make this happen, is the difficulty in assessing the quality and sufficiency of datasets used in training or validating machine learning models (Breck et al., 2019). This is particularly a pain point in the world of computer vision. Our project, VisualInsights will support AI practitioners in the computer vision field to seamlessly and intuitively assess image datasets through an interactive interface. 

## INSTALLATION
Note: Optional items marked below are not needed for demo, data pipeline has been run for demo purposes and output files saved in the proper directories. You can simply skip to "EXECUTION" after cloning the repo (1) and installing the requirements (3).

1. Pull the repository (main branch) to your local git repos to get started.
2. [Optional] Create a virtual environment instead of using base env. If you do not want to create a virtual environment, proceed to step 3.
   <br/>a. In terminal execute this command: 
   ```commandline
   conda create -n visualinsights python=3.8 -y
   ```
   b. once the environment is created, we activate it by executing the following command line.
    ```commandline
    conda activate visualinsights
    ```
3. Install requirements - install the required libraries using the 'requirements.txt' file by executing the command below.
```commandline
pip install -r requirements.txt
```

### Exploration Using Python Code [Optional]
If you have run using jupyter notebooks, you can skip this section.
1. Make sure you have downloaded the nuscenes meta data files & "CAM_FRONT" image samples  to "data/input_data" folder path.
2. Update config parameters in config/config.yaml file - update the nuscenes version & data path for which you want to execute the application. Either mini (v1.0-mini) or the full data (v1.0-trainval_meta)
3. Set the "run_data_loader" and "run_data_processor" flags in main.py as follows:
   1. If it's your first run and you need to load data, extract class names from metadata & save it as a JSON file, and the process the JSON file to generate class distribution information, set both flags to "True."
   2. If you've previously loaded and extracted data, and you have a saved JSON file, you can skip the data loading step by setting the loading flag to "False" and proceed with data processing.
4. Execute the application by running main.py. Make sure your terminal is in the root folder directory.
```commandline
python main.py
```
5. This would generate the csv files in "data/output_data/" directory.

### Exploration Using Jupyter Notebooks [Optional]
Skip to next section if you want to execute the same using python scripts.
<br> **Important**: Make sure you have downloaded the nuscenes meta data files & "CAM_FRONT" image samples to "data/input_data" folder path.
<br> You can find the jupyter notebooks in the "notebooks" folder in the root directory.

1. Run "generate_class_distributions.ipynb" notebook to generate the distribution and counts of object classes in the images.
2. Run "generate_feature_vectors.ipynb" notebook to generate feature vectors for each image and store them in the sqlite db.
3. Run "compute_top_n_similar_images.ipynb" notebook to compute top n similar images for each image using cosine similarity matrix for all images.

#### Clustering Algorithms
You can find all clustering related algorithms run on the images in a jupyter notebook found in in the "notebooks" folder under the name "Cluster Generation and Metadata Generator".
By running the notebook it generates all the data in the following two folders: "clustering_output" and "custering_stats" that are to be used in the UI. To run the notebook you need to run the "imageclusters.ipynb" initially and then use the SQLite database file path as input.

**To avoid heavy computation and time consumption its recommeneded not to run the notebook,as notebook has been run and all necessary files are already allocated.**

## EXECUTION 
To keep a light version of the data, you will first need to untar the data file under data/. To do so, all you need to run is
```commandline
python ui/data_untar.py
```

To launch the Streamlit UI, make sure you have Python version 3.7, then run
```commandline
python -m streamlit run ui/home.py
```

## DEMO VIDEO
Demo video can be found here: https://www.youtube.com/watch?v=Auhm9VxOdgs
