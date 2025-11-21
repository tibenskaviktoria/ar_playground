import trimesh
import pyrender
import numpy as np

def main():
    # Create a Pyrender scene
    scene = pyrender.Scene()

    # Load the Jeep model
    loaded_mesh = trimesh.load("./3D_models/Jeep_Renegade_2016.obj")

    if isinstance(loaded_mesh, trimesh.Scene):
        # If it's a scene, combine all geometries into a single mesh
        for name, mesh in loaded_mesh.geometry.items():
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
            scene.add(pyrender_mesh, name=name)
    else:
        pyrender_mesh = pyrender.Mesh.from_trimesh(loaded_mesh)
        scene.add(pyrender_mesh)

    # Add a directional light to the scene
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4))  # Add light at the default pose

    # Create a camera and add it to the scene
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 4],  # Move the camera 4 units away from the origin
        [0, 0, 0, 1]
    ])
    scene.add(camera, pose=camera_pose)

    # Create a Pyrender Viewer to display the scene
    pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == "__main__":
    main()