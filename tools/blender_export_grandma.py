"""Blender (headless): grandma glTF -> FBX for Mixamo auto-rig. Bench removed, rig removed, textures embedded.
blender -b -P tools/blender_export_grandma.py"""
import bpy, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.join(root, "assets/sketchfab/grandma/scene.gltf"))
for o in list(bpy.data.objects):
    if o.type == "MESH" and any(k in o.data.name for k in ("Cube_002", "Teeth", "Ring", "Necklace")):
        bpy.data.objects.remove(o, do_unlink=True)
for a in bpy.data.armatures:
    a.pose_position = "REST"     # A-pose (rest), not the sitting pose stored in the file
for o in list(bpy.data.objects):
    if o.type == "MESH" and o.name.startswith("Icosphere"):
        bpy.data.objects.remove(o, do_unlink=True)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
for o in meshes:  # bake the rig's pose into each mesh (keeps the dress on her), then drop the rig
    bpy.context.view_layer.objects.active = o
    if o.data.shape_keys:
        o.shape_key_clear()   # facial shape keys block applying the rig
    for m in list(o.modifiers):
        if m.type == "ARMATURE":
            bpy.ops.object.modifier_apply(modifier=m.name)
    mw = o.matrix_world.copy()
    o.parent = None
    o.matrix_world = mw
for o in list(bpy.data.objects):
    if o.type != "MESH":
        bpy.data.objects.remove(o, do_unlink=True)
for o in meshes:   # keep one UV set, decimate for the 10 MB upload limit
    while len(o.data.uv_layers) > 1:
        o.data.uv_layers.remove(o.data.uv_layers[-1])
    d = o.modifiers.new("dec", "DECIMATE")
    d.ratio = 0.35
bpy.ops.object.select_all(action="SELECT")
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.object.join()   # Mixamo's auto-rigger wants one mesh
meshes = [bpy.context.view_layer.objects.active]
dims = [max(v.co[i] for o in meshes for v in o.data.vertices) - min(v.co[i] for o in meshes for v in o.data.vertices) for i in range(3)]
print("DIMS", dims)
out = os.path.join(root, "assets/mixamo/grandma_for_mixamo.fbx")
bpy.ops.export_scene.fbx(filepath=out, path_mode="STRIP", embed_textures=False, apply_scale_options="FBX_SCALE_ALL", use_mesh_modifiers=True)
print("WROTE", out, os.path.getsize(out))
