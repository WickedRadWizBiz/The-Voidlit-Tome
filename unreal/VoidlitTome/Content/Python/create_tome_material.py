"""Build the Tome's materials inside the Unreal Editor.

Run once from the editor's Python console (Edit > Editor Preferences > Python must be enabled):

    exec(open(r"<YourProject>/Plugins/VoidlitTome/Content/Python/create_tome_material.py").read())

or: Tools > Execute Python Script... and pick this file.

Creates, in the plugin's content folder (/VoidlitTome/):
  M_TomeVoxel  the voxel material: Custom nodes calling /Plugin/VoidlitTome/TomeVoxel.ush
  M_TomeSlab   the matte black the voxels vanish into
"""

import unreal

PACKAGE = "/VoidlitTome"
INCLUDE = "/Plugin/VoidlitTome/TomeVoxel.ush"
MATTE = (0.012, 0.012, 0.014)

mel = unreal.MaterialEditingLibrary
assets = unreal.EditorAssetLibrary

TEXTURE_INPUTS = ["StateTex", "ColorTex"]
SCALARS = {
    # name: default
    "TomeTime": 0.0,
    "GridCols": 320.0,
    "GridRows": 720.0,
    "RiseCm": 0.1,
    "RiseSeconds": 0.35,
    "RippleTime": -100.0,
    "RippleStrength": 0.0,
    "Panic": 0.0,
    "Thinking": 0.0,
    "Shudder": 0.0,
    "Lumen": 1.0,
    "Dark": 0.0,
    "Heartbeat": 0.0,
    "Flare": 0.0,
}

COMMON = ("StateTex, StateTexSampler, ColorTex, ColorTexSampler, float2(Col, Row), float2(GridCols, GridRows), "
          "TomeTime")
FXA = "float4(RiseSeconds, RippleTime, RippleStrength, Panic)"
FXB = "float4(Thinking, Shudder, Lumen, Dark)"
FXC = "float4(Heartbeat, Flare, 0, 0)"

CUSTOM_NODES = {
    "WPO": (
        f"return FaceNormalWS * RiseCm * TomeVoxelHeight({COMMON}, {FXA}, {FXB});",
        unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET,
        ["FaceNormalWS", "RiseCm"],
    ),
    "Emissive": (
        f"return TomeVoxelEmissive({COMMON}, {FXA}, {FXB}, {FXC});",
        unreal.MaterialProperty.MP_EMISSIVE_COLOR,
        [],
    ),
    "BaseColor": (
        f"return TomeVoxelBaseColor({COMMON}, {FXA}, {FXB});",
        unreal.MaterialProperty.MP_BASE_COLOR,
        [],
    ),
}


def fresh_material(name):
    path = f"{PACKAGE}/{name}"
    if assets.does_asset_exist(path):
        assets.delete_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    return tools.create_asset(name, PACKAGE, unreal.Material, unreal.MaterialFactoryNew())


def constant(mat, value, x, y):
    node = mel.create_material_expression(mat, unreal.MaterialExpressionConstant, x, y)
    node.set_editor_property("r", value)
    return node


def build_slab():
    mat = fresh_material("M_TomeSlab")
    color = mel.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -400, 0)
    color.set_editor_property("constant", unreal.LinearColor(*MATTE, 1.0))
    mel.connect_material_property(color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    mel.connect_material_property(constant(mat, 0.82, -400, 150), "", unreal.MaterialProperty.MP_ROUGHNESS)
    mel.connect_material_property(constant(mat, 0.3, -400, 250), "", unreal.MaterialProperty.MP_SPECULAR)
    mel.recompile_material(mat)
    assets.save_loaded_asset(mat)
    return mat


def build_voxel():
    mat = fresh_material("M_TomeVoxel")
    mat.set_editor_property("used_with_instanced_static_meshes", True)

    sources = {}
    y = -600
    for name in TEXTURE_INPUTS:
        node = mel.create_material_expression(mat, unreal.MaterialExpressionTextureObjectParameter, -1400, y)
        node.set_editor_property("parameter_name", "TomeState" if name == "StateTex" else "TomeColor")
        node.set_editor_property("texture", unreal.load_asset("/Engine/EngineResources/DefaultTexture"))
        node.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
        sources[name] = node
        y += 160
    for i, (name, default) in enumerate(SCALARS.items()):
        node = mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -1400, y + i * 70)
        node.set_editor_property("parameter_name", name)
        node.set_editor_property("default_value", default)
        sources[name] = node
    normal = mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -1100, -700)
    normal.set_editor_property("parameter_name", "FaceNormalWS")
    normal.set_editor_property("default_value", unreal.LinearColor(1.0, 0.0, 0.0, 0.0))
    sources["FaceNormalWS"] = normal
    for index, name in enumerate(("Col", "Row")):
        node = mel.create_material_expression(mat, unreal.MaterialExpressionPerInstanceCustomData, -1100,
                                              -500 + index * 120)
        node.set_editor_property("data_index", index)
        sources[name] = node

    shared = TEXTURE_INPUTS + ["Col", "Row", "GridCols", "GridRows", "TomeTime"] + list(SCALARS)[4:]
    for row, (label, (code, prop, extra)) in enumerate(CUSTOM_NODES.items()):
        node = mel.create_material_expression(mat, unreal.MaterialExpressionCustom, -500, -400 + row * 350)
        node.set_editor_property("description", f"Tome {label}")
        node.set_editor_property("code", code)
        node.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
        node.set_editor_property("include_file_paths", [INCLUDE])
        names = list(dict.fromkeys(shared + extra))
        inputs = []
        for name in names:
            ci = unreal.CustomInput()
            ci.set_editor_property("input_name", name)
            inputs.append(ci)
        node.set_editor_property("inputs", inputs)
        for name in names:
            mel.connect_material_expressions(sources[name], "", node, name)
        mel.connect_material_property(node, "", prop)

    mel.connect_material_property(constant(mat, 0.82, -500, 700), "", unreal.MaterialProperty.MP_ROUGHNESS)
    mel.connect_material_property(constant(mat, 0.3, -500, 800), "", unreal.MaterialProperty.MP_SPECULAR)
    mel.recompile_material(mat)
    assets.save_loaded_asset(mat)
    return mat


build_slab()
build_voxel()
unreal.log("Voidlit Tome: created /VoidlitTome/M_TomeVoxel and /VoidlitTome/M_TomeSlab")
