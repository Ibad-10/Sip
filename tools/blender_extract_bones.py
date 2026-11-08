"""Blender (headless): sample bone world positions from the Mixamo 'Sitting Drinking' FBX.
blender -b -P tools/blender_extract_bones.py"""
import bpy, json, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=os.path.join(root, "assets/mixamo/sitting_drinking.fbx"))
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
act = arm.animation_data.action
f0, f1 = (int(x) for x in act.frame_range)
names = ["Hips", "Spine2", "Neck", "Head", "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand",
         "LeftUpLeg", "RightUpLeg"]
out = {"frames": []}
for f in range(f0, f1 + 1, 3):
    bpy.context.scene.frame_set(f)
    pos = {}
    for n in names:
        b = arm.pose.bones.get("mixamorig:" + n) or next((pb for pb in arm.pose.bones if pb.name.endswith(n)), None)
        if b is not None:
            pos[n] = list(arm.matrix_world @ b.head)
    out["frames"].append({"f": f, "pos": pos})
json.dump(out, open(os.path.join(root, "assets/mixamo/sitting_drinking_bones.json"), "w"))
print("FRAMES", f0, f1, len(out["frames"]), sorted(out["frames"][0]["pos"]))
