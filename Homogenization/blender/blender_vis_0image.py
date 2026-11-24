import bpy
import numpy as np
from mathutils import Vector
import random
import math

# ────────────────────────────────────────────
# user settings
# ────────────────────────────────────────────
NPY_PATH   = "/Users/justint/Library/CloudStorage/OneDrive-Personal/Desktop/Academic Stuff/Arias Research/Polymers/2025/Paper_Polymers_2025/new/Homogenization/blender"
BASE_COLORS = [
    (1.0, 0.0, 0.0, 1.0),  # red
    (0.0, 0.65, 0.55, 1.0), # teal
]
THRESHOLD   = 0.5      # “Greater-Than” cutoff
OFFSET_STEP = 0.001    # spacing between separate blocks
N_BLOCKS    = 2        # how many density fields to import
# ────────────────────────────────────────────

# ────────────────────────────────────────────
# lattice & periodic images settings
# ────────────────────────────────────────────
# Define your unit‐cell lattice vectors as rows of a 3×3 array:
#    [ a_x, a_y, a_z ]
#    [ b_x, b_y, b_z ]
#    [ c_x, c_y, c_z ]
# Replace the example below with your true lattice:
LATTICE = np.array([
    [1, 0, 0],    # a‐vector
    [0, 1, 0],    # b‐vector
    [0, 0, 1],    # c‐vector
], dtype=float)

# number of periodic “copies” in each direction (–N_IMAGES…+N_IMAGES)
N_IMAGES = 0

# Translate the triply-periodic structure in fractional (lattice) coordinates
# Per-block fractional shifts
FRAC_SHIFTS = [
    np.array([0.0, 0.0, 0.0]),  # block 0: no shift
    np.array([0.0, 0.0, 0.0]),  # block 1: shifted
]

WRAP_SHIFT = True  # keep points inside [0,1) after shifting
# ────────────────────────────────────────────


# ╭──────────────────────────────────────────╮
# │ 0. load density fields from .npy file  │
# ╰──────────────────────────────────────────╯
rho_grids = np.load(NPY_PATH)
if rho_grids.ndim != 4:
    raise ValueError(f"Expected array shape (N, gx, gy, gz); got {rho_grids.shape}")
N_BLOCKS = min(N_BLOCKS, len(rho_grids))

gx, gy, gz = rho_grids[0].shape


# ╭──────────────────────────────────────────╮
# │ helper: create Geo‐Nodes I/O sockets     │
# ╰──────────────────────────────────────────╯
def ensure_geometry_sockets(node_group):
    if hasattr(node_group, "inputs"):
        if "Geometry" not in node_group.inputs:
            node_group.inputs.new("NodeSocketGeometry", "Geometry")
        if "Geometry" not in node_group.outputs:
            node_group.outputs.new("NodeSocketGeometry", "Geometry")
    else:
        iface = node_group.interface
        has_in  = any(s.name == "Geometry" and s.in_out == 'INPUT'  for s in iface.items_tree)
        has_out = any(s.name == "Geometry" and s.in_out == 'OUTPUT' for s in iface.items_tree)
        if not has_in:
            iface.new_socket(name="Geometry", in_out='INPUT',  socket_type='NodeSocketGeometry')
        if not has_out:
            iface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')


# ╭──────────────────────────────────────────╮
# │ main loop – one mesh/object per block    │
# ╰──────────────────────────────────────────╯
for i in range(N_BLOCKS):

    rho = rho_grids[i]
    verts, dens = [], []
    # pick shift for this block (default = no shift if list too short)
    shift = FRAC_SHIFTS[i] if i < len(FRAC_SHIFTS) else np.zeros(3)

    # build all points (including periodic images)
    for ix in range(gx):
        for iy in range(gy):
            for iz in range(gz):
                v = float(rho[ix, iy, iz])
                if v <= 0.0:
                    continue

                # fractional coord in [0,1]^3
                f = np.array([ix/(gx-1), iy/(gy-1), iz/(gz-1)], dtype=float)

                # per-block shift
                f = f + shift
                if WRAP_SHIFT:
                    f = f % 1.0

                # tile periodic images
                for px in range(-N_IMAGES, N_IMAGES+1):
                    for py in range(-N_IMAGES, N_IMAGES+1):
                        for pz in range(-N_IMAGES, N_IMAGES+1):
                            f_shift = f + np.array([px, py, pz])
                            # real‐space: sum_i f_shift[i] * LATTICE[i]
                            r = f_shift.dot(LATTICE)
                            verts.append((r[0], r[1], r[2]))
                            dens.append(v)

    # create mesh & object
    mesh = bpy.data.meshes.new(f"RhoMesh_{i}")
    mesh.from_pydata(verts, [], [])
    mesh.update()

    # density attribute
    att = mesh.attributes.new(name="density", type='FLOAT', domain='POINT')
    for idx, value in enumerate(dens):
        att.data[idx].value = value

    obj = bpy.data.objects.new(f"RhoObj_{i}", mesh)
    obj.location = Vector((i * OFFSET_STEP, 0.0, 0.0))
    bpy.context.collection.objects.link(obj)


    # ── material setup ────────────────────────
    col = BASE_COLORS[i] if i < len(BASE_COLORS) else (random.random(), random.random(), random.random(), 1.0)
    mat = bpy.data.materials.new(f"RhoMat_{i}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    # nodes
    n_out  = nt.nodes.new("ShaderNodeOutputMaterial"); n_out.location  = (500, 0)
    n_mix  = nt.nodes.new("ShaderNodeMixShader");       n_mix.location  = (300, 0)
    n_prin = nt.nodes.new("ShaderNodeBsdfPrincipled");  n_prin.location = ( 80, 80)
    n_prin.inputs["Base Color"].default_value = col
    n_prin.inputs["Roughness"].default_value   = 1.0
    n_tran = nt.nodes.new("ShaderNodeBsdfTransparent"); n_tran.location = ( 80,-80)
    n_attr = nt.nodes.new("ShaderNodeAttribute");       n_attr.location = (-300, 0)
    n_attr.attribute_name = "density"
    n_gt   = nt.nodes.new("ShaderNodeMath");            n_gt.location   = (-100, 0)
    n_gt.operation = 'GREATER_THAN'
    n_gt.inputs[1].default_value = THRESHOLD

    # links
    nt.links.new(n_attr.outputs["Fac"],   n_gt.inputs[0])
    nt.links.new(n_tran.outputs["BSDF"],  n_mix.inputs[1])
    nt.links.new(n_prin.outputs["BSDF"],  n_mix.inputs[2])
    nt.links.new(n_gt.outputs["Value"],   n_mix.inputs["Fac"])
    nt.links.new(n_mix.outputs["Shader"], n_out.inputs["Surface"])

    obj.data.materials.append(mat)


    # ── geometry‐nodes modifier ─────────────────
    mod = obj.modifiers.new("GeoNodes", type='NODES')
    if mod.node_group is None:
        mod.node_group = bpy.data.node_groups.new(f"Geo_Rho_{i}", 'GeometryNodeTree')
    ng = mod.node_group
    ensure_geometry_sockets(ng)
    ng.nodes.clear()

    n_gin  = ng.nodes.new("NodeGroupInput");         n_gin.location  = (-400, 0)
    n_mtp  = ng.nodes.new("GeometryNodeMeshToPoints"); n_mtp.location = (-100, 0)
    n_mtp.mode = 'VERTICES'
    n_mtp.inputs['Radius'].default_value = 0.02

    n_setm = ng.nodes.new("GeometryNodeSetMaterial"); n_setm.location = (150, 0)
    n_setm.inputs["Material"].default_value = mat
    n_gout = ng.nodes.new("NodeGroupOutput");        n_gout.location = (400, 0)

    ng.links.new(n_gin.outputs["Geometry"], n_mtp.inputs["Mesh"])
    ng.links.new(n_mtp.outputs["Points"],   n_setm.inputs["Geometry"])
    ng.links.new(n_setm.outputs["Geometry"],n_gout.inputs["Geometry"])


# ── position camera & light ─────────────────
cam = bpy.data.objects.get("Camera")
if cam:
    cam.location = Vector((4.45, -3.34, 1.93))
    cam.rotation_mode = "XYZ"
    cam.rotation_euler = (
        math.radians(73.93),
        math.radians( 0.00),
        math.radians(47.17),
    )

# ── Sun light (aims straight down) ────────────────────────────────────────────
def ensure_sun_light(
    name="Light",
    strength=9.0,
    angle_deg=45.4,
    color=(1.0, 1.0, 1.0),
    place_at=Vector((6.0, -4.0, 5.0)),  # location is arbitrary for Sun
    aim_down=True                        # << new: aim straight down if True
):
    # Get or create a light object
    lt_obj = bpy.data.objects.get(name)
    if lt_obj is None or lt_obj.type != 'LIGHT':
        light_data = bpy.data.lights.new(name=name, type='SUN')
        lt_obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.collection.objects.link(lt_obj)
    else:
        if lt_obj.data.type != 'SUN':
            lt_obj.data.type = 'SUN'

    # Settings
    sun = lt_obj.data
    sun.color = color
    sun.energy = strength
    sun.angle = math.radians(angle_deg)  # UI is degrees; API is radians
    sun.use_shadow = True

    # Place it (position doesn't affect a Sun's illumination)
    lt_obj.location = place_at

    # --- Aim it ---
    lt_obj.rotation_mode = 'XYZ'
    if aim_down:
        # Make the light look straight down the world -Z axis
        lt_obj.rotation_euler = (0.0, 0.0, 0.0)
        # (For safety if the object is in quaternion mode elsewhere, you could also do:)
        # lt_obj.rotation_quaternion = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    else:
        # Old behavior: aim at some target point
        aim_at = Vector((0.0, 0.0, 0.0))
        lt_obj.rotation_euler = (aim_at - lt_obj.location).to_track_quat('-Z', 'Y').to_euler()

    return lt_obj

# After the camera setup:
ensure_sun_light(
    name="Light",
    strength=9.0,
    angle_deg=45.4,
    color=(1.0, 1.0, 1.0),
    place_at=Vector((5.4734, -4.1871, 5.0312)),
    aim_down=True   # << key line
)


# ── camera-only white backdrop plane (no lighting impact) ──────────────────────
def add_camera_backdrop(
    cam_obj,
    distance=10.0,           # how far in front of the camera the backdrop sits
    strength=10.0,           # emission brightness for the white
    name="CameraBackdrop"
):
    if cam_obj is None or cam_obj.type != 'CAMERA':
        print("No camera found; skipping backdrop.")
        return

    scene   = bpy.context.scene
    render  = scene.render
    res_x   = render.resolution_x * render.pixel_aspect_x
    res_y   = render.resolution_y * render.pixel_aspect_y
    aspect  = (res_x / res_y) if res_y > 0 else 1.0

    camd = cam_obj.data
    # Use vertical FOV for robust sizing (independent of sensor fit)
    fov_y = getattr(camd, "angle_y", None)
    if fov_y is None:
        # Fallback: infer from angle_x and aspect if needed
        angle_x = getattr(camd, "angle_x", camd.angle)
        fov_y = 2.0 * math.atan(math.tan(angle_x * 0.5) / aspect)

    # Compute plane width/height so it perfectly fills the camera frame at 'distance'
    half_h = math.tan(0.5 * fov_y) * distance
    half_w = half_h * aspect
    width, height = 2.0 * half_w, 2.0 * half_h

    # Make/reuse the plane
    plane = bpy.data.objects.get(name)
    if plane is None:
        mesh = bpy.data.meshes.new(name + "_Mesh")
        plane = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(plane)
        # Build a unit plane (2x2) so scale maps cleanly
        bm = bpy.data.meshes.new_from_object(
            bpy.data.objects.new("temp", bpy.data.meshes.new("temp_mesh"))
        )  # we'll replace with a proper primitive below
        # Simpler: use built-in operator to add a plane, then rename & reuse data
        bpy.ops.mesh.primitive_plane_add(size=40.0, enter_editmode=False, align='WORLD')
        tmp = bpy.context.active_object
        plane.data = tmp.data
        bpy.data.objects.remove(tmp, do_unlink=True)
        plane.name = name
        plane.data.name = name + "_Mesh"

    # Parent to camera so it follows framing
    plane.parent = cam_obj
    plane.matrix_parent_inverse = cam_obj.matrix_world.inverted()

    # Place the plane straight in front of the camera and align it to the camera
    cam_mat = cam_obj.matrix_world
    cam_forward = (cam_mat.to_quaternion() @ Vector((0, 0, -1)))  # camera looks along -Z
    plane.location = cam_obj.location + cam_forward * distance
    plane.rotation_mode = 'QUATERNION'
    plane.rotation_quaternion = cam_mat.to_quaternion()

    # Scale so it fills the frame exactly at the chosen distance
    # Default plane is 2x2, so scale by half the desired width/height
    plane.scale = Vector((width * 0.5, height * 0.5, 1.0))

    # Material: pure white emission so it renders white, but won’t affect scene if we disable GI
    mat = bpy.data.materials.get(name + "_Mat")
    if mat is None:
        mat = bpy.data.materials.new(name + "_Mat")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        n_out = nt.nodes.new("ShaderNodeOutputMaterial"); n_out.location = (200, 0)
        n_em  = nt.nodes.new("ShaderNodeEmission");       n_em.location  = (  0, 0)
        n_em.inputs["Color"].default_value = (1, 1, 1, 1)  # white
        n_em.inputs["Strength"].default_value = strength
        nt.links.new(n_em.outputs["Emission"], n_out.inputs["Surface"])
    plane.data.materials.clear()
    plane.data.materials.append(mat)

    # Make sure it does NOT affect lighting/shadows (Cycles path)
    if hasattr(plane, "cycles_visibility"):
        vis = plane.cycles_visibility
        vis.camera       = True
        vis.diffuse      = False
        vis.glossy       = False
        vis.transmission = False
        vis.scatter      = False
        vis.shadow       = False

    # Also disable shadow cast/receive where applicable
    if hasattr(plane, "visible_shadow"):
        plane.visible_shadow = False
    if hasattr(plane, "visible_diffuse"):
        plane.visible_diffuse = False

    # For Eevee, prevent lighting impact by disabling light probes & ambient occlusion influence
    if hasattr(plane, "eevee"):
        try:
            plane.eevee.lightgroup = ""
        except Exception:
            pass

    # Ensure it draws behind your object: push it far enough & disable selection overlays in render
    plane.hide_render = False
    plane.hide_set(False)
    return plane

# (1) Turn OFF compositor nodes to avoid accidental black outputs from a bad tree
bpy.context.scene.use_nodes = False

# (2) Keep world lighting intact
bpy.context.scene.render.film_transparent = False  # we’re not compositing; leave the film opaque

# (3) Create the backdrop
cam_obj = bpy.data.objects.get("Camera")
add_camera_backdrop(cam_obj, distance=40.0, strength=10.0, name="CameraBackdrop")


print(f"✅ Finished: created {N_BLOCKS} rho_grid objects "
      f"({gx}×{gy}×{gz} voxels each, "
      f"{(2*N_IMAGES+1)**3} periodic images).")
