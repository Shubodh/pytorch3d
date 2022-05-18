# original vids
python make_video.py --scene_id 01 --on_ada ada --ref_or_query ref 
python make_video.py --scene_id 03 --on_ada ada --ref_or_query ref 
python make_video.py --scene_id 05 --on_ada ada --ref_or_query ref 
python make_video.py --scene_id 07 --on_ada ada --ref_or_query ref 
python make_video.py --scene_id 09 --on_ada ada --ref_or_query ref 

python make_video.py --scene_id 01 --on_ada ada --ref_or_query query
python make_video.py --scene_id 03 --on_ada ada --ref_or_query query
python make_video.py --scene_id 05 --on_ada ada --ref_or_query query
python make_video.py --scene_id 07 --on_ada ada --ref_or_query query
python make_video.py --scene_id 09 --on_ada ada --ref_or_query query

# rendered vids
python make_video.py --scene_id 01 --on_ada ada --ref_or_query ref --save_rendered_vid
python make_video.py --scene_id 03 --on_ada ada --ref_or_query ref --save_rendered_vid
python make_video.py --scene_id 05 --on_ada ada --ref_or_query ref --save_rendered_vid
python make_video.py --scene_id 07 --on_ada ada --ref_or_query ref --save_rendered_vid
python make_video.py --scene_id 09 --on_ada ada --ref_or_query ref --save_rendered_vid

python make_video.py --scene_id 01 --on_ada ada --ref_or_query query --save_rendered_vid
python make_video.py --scene_id 03 --on_ada ada --ref_or_query query --save_rendered_vid
python make_video.py --scene_id 05 --on_ada ada --ref_or_query query --save_rendered_vid
python make_video.py --scene_id 07 --on_ada ada --ref_or_query query --save_rendered_vid
python make_video.py --scene_id 09 --on_ada ada --ref_or_query query --save_rendered_vid
