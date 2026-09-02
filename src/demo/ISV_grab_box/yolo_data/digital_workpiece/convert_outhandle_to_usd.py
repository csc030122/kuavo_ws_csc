#!/usr/bin/env python3
"""Convert outhandle.stp to a monolithic USD with an Isaac Sim plastic material."""

from pathlib import Path

from isaacsim import SimulationApp


SOURCE = Path(__file__).with_name("outhandle.stp").resolve()
OUTPUT = Path(__file__).with_name("outhandle.usd").resolve()
BASE_COLOR_SRGB_8BIT = (35, 35, 35)


def srgb_to_linear(channel_8bit):
    """Convert an 8-bit sRGB channel to the linear value expected by PBR shaders."""
    value = channel_8bit / 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


BASE_COLOR = tuple(srgb_to_linear(channel) for channel in BASE_COLOR_SRGB_8BIT)


def create_material(stage):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, "/Looks/BlackSemiMatteInjectionPlastic")
    material_prim = material.GetPrim()
    material_prim.SetCustomDataByKey("description", "Deep-black non-metallic semi-matte injection-molded plastic")
    material_prim.SetCustomDataByKey("baseColor_sRGB_8bit", ", ".join(map(str, BASE_COLOR_SRGB_8BIT)))
    material_prim.SetCustomDataByKey("baseColor_linear", ", ".join(f"{value:.8f}" for value in BASE_COLOR))

    # Isaac Sim / RTX preferred render context: OmniPBR MDL.
    mdl_shader = UsdShade.Shader.Define(stage, "/Looks/BlackSemiMatteInjectionPlastic/OmniPBR")
    mdl_shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    mdl_shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    mdl_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*BASE_COLOR))
    mdl_shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    mdl_shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.55)
    mdl_shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(False)
    mdl_shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(1.0)
    mdl_shader.CreateInput("enable_opacity_texture", Sdf.ValueTypeNames.Bool).Set(False)
    mdl_output = mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_output)

    # Portable fallback for renderers that do not support MDL.
    preview_shader = UsdShade.Shader.Define(stage, "/Looks/BlackSemiMatteInjectionPlastic/UsdPreviewSurface")
    preview_shader.CreateIdAttr("UsdPreviewSurface")
    preview_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*BASE_COLOR))
    preview_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    preview_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    preview_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    preview_shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
    preview_output = preview_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(preview_output)
    return material


def bind_material_and_report(output_path):
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(output_path))
    if not stage:
        raise RuntimeError(f"Could not open converted USD: {output_path}")

    material = create_material(stage)
    mesh_count = 0
    face_count = 0
    point_count = 0

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            mesh_count += 1
            face_count += len(mesh.GetFaceVertexCountsAttr().Get() or [])
            point_count += len(mesh.GetPointsAttr().Get() or [])
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                UsdShade.Tokens.strongerThanDescendants,
            )

    if mesh_count == 0:
        raise RuntimeError("CAD conversion produced no meshes")

    from canonicalize_workpiece_usd import canonicalize_stage

    canonicalize_stage(stage, Path(output_path).stem)

    stage.GetRootLayer().Save()
    return mesh_count, face_count, point_count


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    app = SimulationApp({"headless": True})
    try:
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("omni.kit.converter.hoops_core", True)
        app.update()

        import omni.converter.hoops
        from omni.kit.converter.hoops_core import HoopsOptions

        options = HoopsOptions()
        options.instancingStyle = omni.converter.hoops.InstancingStyle.eNone
        options.compositionStyle = omni.converter.hoops.CompositionStyle.eNone
        options.filterStyle = omni.converter.hoops.FilterStyle.eOmit
        options.tessLOD = 3
        options.accurateSurfaceCurvatures = True
        options.accurateTessellation = False
        options.dedup = True
        options.useMaterials = False
        options.useNormals = True
        options.omitHiddenOnLoad = True
        options.convertCurves = False
        options.convertMetadata = True
        options.convertPhysicsData = False

        converter = omni.converter.hoops.Converter(options)
        error_code, error_message = converter.convert(str(SOURCE), str(OUTPUT), {})
        if error_code != 0:
            raise RuntimeError(f"HOOPS conversion failed ({error_code}): {error_message}")

        meshes, faces, points = bind_material_and_report(OUTPUT)
        print(f"OUTPUT={OUTPUT}")
        print(f"MESHES={meshes}")
        print(f"FACES={faces}")
        print(f"POINTS={points}")
        print(f"BASE_COLOR={BASE_COLOR}")
        print("METALLIC=0.0")
        print("ROUGHNESS=0.55")
        print("OPACITY=1.0")
        print("CLEARCOAT=0.0 (USD Preview Surface); disabled/not exposed by OmniPBR")
    finally:
        app.close()


if __name__ == "__main__":
    main()
