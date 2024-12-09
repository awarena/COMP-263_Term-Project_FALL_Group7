# %%
from pathlib import Path

# %%
path = Path('frames')
gestures = list(path.iterdir())

# %%
final_json = {}
for path in gestures:
    instances = list(path.iterdir())
    filled = []
    for instance_path in instances:
        if len(list(instance_path.iterdir())) > 0:
            filled.append(instance_path)
            
    if filled:
        for instance_path in filled:
            final_json[f"{path.name}/{instance_path.name}"] = {}
# %%
import pandas as pd
import json
from pathlib import Path

# open WLASL_v0.3.json
df = pd.read_json("WLASL_v0.3.json")

# apply function to each row
def process_row(gloss, instances):
    for instance in instances:
        videoId = instance['video_id']
        bbox = instance['bbox']
        fps = instance['fps']
        frame_start = instance['frame_start']
        frame_end = instance['frame_end']
        split = instance['split']
        if f"{gloss}/{videoId}" in final_json:
            final_json[f"{gloss}/{videoId}"] = {
                "frame_start": frame_start,
                "frame_end": frame_end
            }

        
# apply function to one row
glosses = df['gloss']
instances = df['instances']
for i in range(len(glosses)):
    process_row(glosses[i], instances[i])
# %%
final_json
# %%
for k, v in final_json.items():
    if not v:
        del final_json[k]
        

# %%
with open('gesture_annotations.json', 'w') as f:
    json.dump(final_json, f)
# %%
df = pd.read_json("gesture_annotations.json")
df
# %%
