import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dataset import newface_token, stopface_token, padface_token


def visualize_points(points, vis_path, colors=None):
    if colors is None:
        Path(vis_path).write_text("\n".join(f"v {p[0]} {p[1]} {p[2]} 127 127 127" for p in points))
    else:
        Path(vis_path).write_text("\n".join(f"v {p[0]} {p[1]} {p[2]} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}" for i, p in enumerate(points)))


def tokens_to_vertices(token_sequence, num_tokens):
    try:
        end = token_sequence.index(num_tokens + 1)
    except ValueError:
        end = len(token_sequence)
    token_sequence = token_sequence[:end]
    token_sequence = token_sequence[:(len(token_sequence) // 3) * 3]
    vertices = (np.array(token_sequence).reshape(-1, 3)) / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    return vertices


def visualize_quantized_mesh_vertices(token_sequence, num_tokens, output_path):
    vertices = tokens_to_vertices(token_sequence, num_tokens)
    plot_vertices(vertices, output_path)


def visualize_quantized_mesh_vertices_and_faces(token_sequence_vertex, token_sequence_face, num_tokens, output_path):
    vertices, faces = tokens_to_mesh(token_sequence_vertex, token_sequence_face, num_tokens)
    plot_vertices_and_faces(vertices, faces, output_path)


def plot_vertices(vertices, output_path):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlim(-0.35, 0.35)
    plt.ylim(-0.35, 0.35)
    # Don't mess with the limits!
    plt.autoscale(False)
    ax.set_axis_off()
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='g', s=10)
    ax.set_zlim(-0.35, 0.35)
    ax.view_init(25, -120, 0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")


def plot_vertices_and_faces(vertices, faces, output_path):
    ngons = [[vertices[v, :].tolist() for v in f] for f in faces]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlim(-0.45, 0.45)
    plt.ylim(-0.45, 0.45)
    # Don't mess with the limits!
    plt.autoscale(False)
    ax.set_axis_off()
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='black', s=10)
    polygon_collection = Poly3DCollection(ngons)
    polygon_collection.set_alpha(0.3)
    polygon_collection.set_color('b')
    ax.add_collection(polygon_collection)
    ax.set_zlim(-0.35, 0.35)
    ax.view_init(25, -120, 0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")


def visualize_quantized_mesh_vertices_gif(token_sequence, num_tokens, output_dir):
    vertices = tokens_to_vertices(token_sequence, num_tokens)
    visualize_mesh_vertices_gif(vertices, output_dir)


def visualize_mesh_vertices_gif(vertices, output_dir):
    for i in range(1, len(vertices), 1):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        plt.xlim(-0.35, 0.35)
        plt.ylim(-0.35, 0.35)
        # Don't mess with the limits!
        plt.autoscale(False)
        ax.set_axis_off()
        ax.scatter(vertices[:i, 0], vertices[:i, 1], vertices[:i, 2], c='g', s=10)
        ax.set_zlim(-0.35, 0.35)
        ax.view_init(25, -120, 0)
        plt.tight_layout()
        plt.savefig(output_dir / f"{i:05d}.png")
        plt.close("all")
    create_gif(output_dir, 40, output_dir / "vis.gif")


def visualize_quantized_mesh_vertices_and_faces_gif(token_sequence_vertex, token_sequence_face, num_tokens, output_dir):
    visualize_quantized_mesh_vertices_gif(token_sequence_vertex, num_tokens, output_dir)
    vertices, faces = tokens_to_mesh(token_sequence_vertex, token_sequence_face, num_tokens)
    visualize_mesh_vertices_and_faces_gif(vertices, faces, output_dir)


def visualize_mesh_vertices_and_faces_gif(vertices, faces, output_dir):
    ngons = [[vertices[v, :].tolist() for v in f] for f in faces]
    for i in range(1, len(ngons) + 1, 1):
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")
        plt.xlim(-0.35, 0.35)
        plt.ylim(-0.35, 0.35)
        # Don't mess with the limits!
        plt.autoscale(False)
        ax.set_axis_off()
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='black', s=10)
        polygon_collection = Poly3DCollection(ngons[:i])
        polygon_collection.set_alpha(0.3)
        polygon_collection.set_color('b')
        ax.add_collection(polygon_collection)
        ax.set_zlim(-0.35, 0.35)
        ax.view_init(25, -120, 0)
        plt.tight_layout()
        plt.savefig(output_dir / f"{len(vertices) + i:05d}.png")
        plt.close("all")
    create_gif(output_dir, 40, output_dir / "vis.gif")


def create_gif(folder, fps, output_path):
    collection_rgb = []
    for f in sorted([x for x in folder.iterdir() if x.suffix == ".png" or x.suffix == ".jpg"]):
        img_rgb = np.array(Image.open(f).resize((384, 384)))
        collection_rgb.append(img_rgb)
    clip = ImageSequenceClip(collection_rgb, fps=fps)
    clip.write_gif(output_path, verbose=False, logger=None)


def tokens_to_mesh(vertices_q, face_sequence, num_tokens):
    vertices = (np.array(vertices_q).reshape(-1, 3)) / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    try:
        end = face_sequence.index(stopface_token)
    except ValueError:
        end = len(face_sequence)
    face_sequence = face_sequence[:end]
    face_sequence = [x for x in face_sequence if x != 2]  # remove padding
    faces = []
    current_face = []
    for i in range(len(face_sequence)):
        if face_sequence[i] == newface_token:
            if len(current_face) > 2:
                faces.append(current_face)
            current_face = []
        else:
            current_face.append(face_sequence[i] - 3)
    if len(current_face) != 0:
        faces.append(current_face)
    return vertices, faces


def ngon_to_obj(vertices, faces):
    obj = ""
    for i in range(len(vertices)):
        obj += f"v {vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]}\n"
    for i in range(len(faces)):
        fline = "f"
        for j in range(len(faces[i])):
            fline += f" {faces[i][j] + 1} "
        fline += "\n"
        obj += fline
    return obj


def trisoup_sequence_to_mesh(soup_sequence, num_tokens):
    try:
        end = soup_sequence.index(stopface_token)
    except ValueError:
        end = len(soup_sequence)
    soup_sequence = soup_sequence[:end]
    vertices_q = []
    current_subsequence = []
    for i in range(len(soup_sequence)):
        if soup_sequence[i] == newface_token:
            if len(current_subsequence) >= 9:
                current_subsequence = current_subsequence[:9]
                vertices_q.append(np.array(current_subsequence).reshape(3, 3))
            current_subsequence = []
        elif soup_sequence[i] != padface_token:
            current_subsequence.append(soup_sequence[i] - 3)
    if len(current_subsequence) >= 9:
        current_subsequence = current_subsequence[:9]
        vertices_q.append(np.array(current_subsequence).reshape(3, 3))
    vertices = (np.array(vertices_q).reshape(-1, 3)) / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    faces = np.array(list(range(len(vertices_q) * 3)), dtype=np.int32).reshape(-1, 3)
    return vertices, faces


def ngonsoup_sequence_to_mesh(soup_sequence, num_tokens):
    try:
        end = soup_sequence.index(stopface_token)
    except ValueError:
        end = len(soup_sequence)
    soup_sequence = soup_sequence[:end]
    vertices_q = []
    face_ctr = 0
    faces = []
    current_subsequence = []
    for i in range(len(soup_sequence)):
        if soup_sequence[i] == newface_token:
            current_subsequence = current_subsequence[:len(current_subsequence) // 3 * 3]
            if len(current_subsequence) > 0:
                vertices_q.append(np.array(current_subsequence).reshape(-1, 3))
                faces.append([x for x in range(face_ctr, face_ctr + len(current_subsequence) // 3)])
                face_ctr += (len(current_subsequence) // 3)
            current_subsequence = []
        elif soup_sequence[i] != padface_token:
            current_subsequence.append(soup_sequence[i] - 3)

    current_subsequence = current_subsequence[:len(current_subsequence) // 3 * 3]
    if len(current_subsequence) > 0:
        vertices_q.append(np.array(current_subsequence).reshape(-1, 3))
        faces.append([x for x in range(face_ctr, face_ctr + len(current_subsequence) // 3)])
        face_ctr += (len(current_subsequence) // 3)

    vertices = np.vstack(vertices_q) / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    return vertices, faces


def triangle_sequence_to_mesh(triangles):
    vertices = triangles.reshape(-1, 3)
    faces = np.array(list(range(vertices.shape[0]))).reshape(-1, 3)
    return vertices, faces
