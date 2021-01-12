"""
Generate annotations from cached data

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision

Instructions to generate requried .pkl files based on the v-coco repo
https://github.com/s-gupta/v-coco

(1) Modify vsrl_utils.py script to add object categories into annotations

@@ -132,6 +132,10 @@ def attach_gt_boxes(vsrl_data, coco):
   vsrl_data['role_bbox'] = \
     np.nan*np.zeros((vsrl_data['role_object_id'].shape[0], \
       4*vsrl_data['role_object_id'].shape[1]), dtype=np.float)
+  vsrl_data['obj_category'] = \
+    np.nan*np.zeros((vsrl_data['role_object_id'].shape[0], \
+      vsrl_data['role_object_id'].shape[1]), dtype=np.float)

@@ -139,8 +143,10 @@ def attach_gt_boxes(vsrl_data, coco):
     if has_role.size > 0:
       anns = coco.loadAnns(vsrl_data['role_object_id'][has_role, i].ravel().tolist());
       bbox = np.vstack([np.array(a['bbox']).reshape((1,4)) for a in anns])
+      obj_cate = np.vstack([np.array(a['category_id']).reshape((1)) for a in anns])
       bbox = xyhw_to_xyxy(bbox)
       vsrl_data['role_bbox'][has_role, 4*i:4*(i+1)] = bbox;
+      vsrl_data['obj_category'][has_role, i] = obj_cate[:, 0]

(2) Cache each partition as pickle files using V-COCO.ipynb

    a. Run cells [1] and [3]
    b. Use pickle to save vcoco_all with 'wb' option
    c. Change the partition name e.g. vcoco_train -> vcoco_val and repeat a. and b.
"""

import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm

INTERACTIONS = [
    'hold obj', 'sit instr', 'ride instr', 'look obj', 'hit instr', 'hit obj', 'eat obj',
    'eat instr', 'jump instr', 'lay instr', 'talk_on_phone instr', 'carry obj', 'throw obj',
    'catch obj', 'cut instr', 'cut obj', 'work_on_computer instr', 'ski instr', 'surf instr',
    'skateboard instr', 'drink instr', 'kick obj', 'read obj', 'snowboard instr'
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', required=True, type=str,
                        help="Path to the pickle file with annotations for a partition")
    parser.add_argument('--partition', required=True, type=int,
                        help="Dataset partition the pickle file corresponds to. " \
                        + "Use 0 for training set, 1 for validation set and 2 for test set")
    args = parser.parse_args()

    fname = args.pickle
    partition = args.partition
    assert partition in [0, 1, 2], "Incorrect partition number"

    with open(fname, 'rb') as f:
        a = pickle.load(f, encoding='latin1')

    unique_im_id = np.unique(a[0]['image_id']).tolist()
    nimage = len(unique_im_id)

    def anno_template(image_id, partition):
        if partition in [0, 1]:
            prefix = 'COCO_train2014'
        else:
            prefix = 'COCO_val2014'
        return dict(
            boxes_h=[],
            boxes_o=[],
            actions=[],
            objects=[],
            file_name=prefix+'_{}.jpg'.format(str(image_id).zfill(12))
        )
    anno = [anno_template(unique_im_id[i], partition) for i in range(nimage)]

    for data in tqdm(a):
        # Remove class point_instr
        if data['action_name'] == 'point':
            continue
        num_classes = len(data['role_name']) - 1
        # Skip actions that do not involve objects. Actions that involve two super
        # categories (object and instrument) are counted twice
        for i in range(num_classes):
            interaction_name = ' '.join(
                [data['action_name'], data['role_name'][i + 1]]
            )
            idx = INTERACTIONS.index(interaction_name)
            
            keep = np.where(data['label'])[0]
            for j in keep:
                image_id = data['image_id'][j]
                k = unique_im_id.index(image_id)
                bh = data['role_bbox'][j, :4]
                bo = data['role_bbox'][j, (i + 1) * 4: (i + 2) * 4]
                # Skip when the object box is not annotated
                if np.isnan(bo).any():
                    continue
                anno[k]['boxes_h'].append(bh.tolist())
                anno[k]['boxes_o'].append(bo.tolist())
                anno[k]['actions'].append(idx)
                anno[k]['objects'].append(data['obj_category'][j, i + 1])

    with open(fname.replace('.pkl', '.json'), 'w') as f:
        json.dump(dict(
            annotations=anno,
            classes=INTERACTIONS,
            images=unique_im_id
        ), f)