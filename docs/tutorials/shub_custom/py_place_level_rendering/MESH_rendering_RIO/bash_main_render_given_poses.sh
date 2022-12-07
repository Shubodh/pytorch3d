#echo "scene 01 query"
#python main_render_given_poses.py --scene_id 01 --ref_or_query query 
#cp /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene01_ROI_with_QOI/query/camera.yaml /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene01_RRI_with_QRI/query/
#echo "scene 03 query"
#python main_render_given_poses.py --scene_id 03 --ref_or_query query 
#cp /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene03_ROI_with_QOI/query/camera.yaml /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene03_RRI_with_QRI/query/
#echo "scene 05 query"
#python main_render_given_poses.py --scene_id 05 --ref_or_query query 
#cp /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene05_ROI_with_QOI/query/camera.yaml /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene05_RRI_with_QRI/query/
#echo "scene 07 query"
#python main_render_given_poses.py --scene_id 07 --ref_or_query query 
#cp /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene07_ROI_with_QOI/query/camera.yaml /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene07_RRI_with_QRI/query/
#echo "scene 09 query"
#python main_render_given_poses.py --scene_id 09 --ref_or_query query 
#cp /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene09_ROI_with_QOI/query/camera.yaml /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene09_RRI_with_QRI/query/

#echo "scene 01 ref"
#python main_render_given_poses.py --scene_id 01 --ref_or_query ref 
#cp /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene01_ROI_with_QOI/database/cutouts/camera.yaml /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene01_RRI_with_QRI/database/cutouts/
#echo "scene 03 ref"
#python main_render_given_poses.py --scene_id 03 --ref_or_query ref 
#cp /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene03_ROI_with_QOI/database/cutouts/camera.yaml /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene03_RRI_with_QRI/database/cutouts/
echo "scene 05 ref"
python main_render_given_poses.py --scene_id 05 --ref_or_query ref 
cp /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene05_ROI_with_QOI/database/cutouts/camera.yaml /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene05_RRI_with_QRI/database/cutouts/
#echo "scene 07 ref"
#python main_render_given_poses.py --scene_id 07 --ref_or_query ref 
#cp /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene07_ROI_with_QOI/database/cutouts/camera.yaml /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene07_RRI_with_QRI/database/cutouts/
#echo "scene 09 ref"
#python main_render_given_poses.py --scene_id 09 --ref_or_query ref 
#cp /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene09_ROI_with_QOI/database/cutouts/camera.yaml /scratch/saishubodh/InLoc_like_RIO10/sampling10/scene09_RRI_with_QRI/database/cutouts/
