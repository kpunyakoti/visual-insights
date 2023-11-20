#Imports
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from pyquaternion import Quaternion
import numpy as np
import tqdm

def informuser(msg, msgtype = "STATUS"):
    if(msgtype == "STATUS"):
        print(msg)
     

nusc = NuScenes(version='v1.0-mini', dataroot='D:\projects\dva\data', verbose=True)
informuser("Finished Loading Dataset...")
informuser("Analysing Annotations...")
camera_name = 'CAM_FRONT'
image_data_dict = dict()
annotation_ids = []
# Iterate on each scene
for scene in tqdm.tqdm(nusc.scene):
    #Iterate on sample
    token = scene['first_sample_token']
    while token != '':
        sample = nusc.get('sample', token)
        camera_token = sample['data'][camera_name]
        camera_data = nusc.get('sample_data', camera_token)
        annotation_list = sample['anns']
        
        # Get the annotations for this sample
        filename, boxes, camera_intrinsic = nusc.get_sample_data(camera_token, selected_anntokens=annotation_list)
        
        #If this same was not there in the list already add an empty array to hold the annotations
        if camera_token not in image_data_dict:
            image_data_dict[camera_token] = {'iID'      : camera_token,
                                             'filePath' : filename,
                                             'annotations' : []}
        
        #For each of the annotations in that sample
        for box in boxes:
            corners = view_points(box.corners(), np.array(camera_intrinsic), normalize=True)[:2, :]
            image_data_dict[camera_token]['annotations'].append({'name': box.name, 
                                                                  'corners': corners.tolist()})
        #Get the next token
        token = sample['next']

        if box.name not in annotation_ids:
            annotation_ids.append(box.name)

informuser("Analysing Annotations Completed.")

print (annotation_ids)




