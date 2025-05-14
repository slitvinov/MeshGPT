import random
import omegaconf
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torch.utils.data
import pickle
from numpy import random
import trimesh
import torch_scatter

from dataset import get_shifted_sequence
from dataset.triangles import create_feature_stack
from trainer import create_conv_batch, get_rvqvae_v0_encoder_vq, get_rvqvae_v1_encoder_vq
from util.misc import normalize_vertices, scale_vertices
from util.visualization import triangle_sequence_to_mesh


class QuantizedSoup(Dataset):

    def __init__(self, config, split, scale_augment):
        super().__init__()
        data_path = Path(config.dataset_root)
        self.block_size = config.block_size
        self.vq_depth = config.embed_levels
        self.vq_is_shared = config.embed_share
        self.vq_num_codes_per_level = config.n_embed
        self.cached_vertices = []
        self.cached_faces = []
        self.names = []
        self.num_tokens = config.num_tokens - 3
        self.scale_augment = scale_augment
        with open(data_path, 'rb') as fptr:
            data = pickle.load(fptr)
            if config.only_chairs:
                for s in ['train', 'val']:
                    data[f'vertices_{s}'] = [data[f'vertices_{s}'][i] for i in range(len(data[f'vertices_{s}'])) if data[f'name_{s}'][i].split('_')[0] == '03001627']
                    data[f'faces_{s}'] = [data[f'faces_{s}'][i] for i in range(len(data[f'faces_{s}'])) if data[f'name_{s}'][i].split('_')[0] == '03001627']
                    data[f'name_{s}'] = [data[f'name_{s}'][i] for i in range(len(data[f'name_{s}'])) if data[f'name_{s}'][i].split('_')[0] == '03001627']
            if not config.overfit:
                self.names = data[f'name_{split}']
                self.cached_vertices = data[f'vertices_{split}']
                self.cached_faces = data[f'faces_{split}']
            else:
                multiplier = 16 if split == 'val' else 512
                self.names = data[f'name_train'][:1] * multiplier
                self.cached_vertices = data[f'vertices_train'][:1] * multiplier
                self.cached_faces = data[f'faces_train'][:1] * multiplier

        print(len(self.cached_vertices), "meshes loaded")

        max_inner_face_len = 0
        for i in range(len(self.cached_vertices)):
            self.cached_vertices[i] = np.array(self.cached_vertices[i])
            for j in range(len(self.cached_faces[i])):
                max_inner_face_len = max(max_inner_face_len, len(self.cached_faces[i][j]))
        print('Longest inner face sequence', max_inner_face_len)
        assert max_inner_face_len == 3, "Only triangles are supported"

        self.start_sequence_token = 0
        self.end_sequence_token = 1
        self.pad_face_token = 2
        self.padding = int(config.padding * self.block_size)
        self.indices = []

        max_face_sequence_len = 0
        min_face_sequence_len = 1e7
        for i in range(len(self.cached_faces)):
            sequence_len = len(self.cached_faces[i]) * self.vq_depth + 1 + 1
            max_face_sequence_len = max(max_face_sequence_len, sequence_len)
            min_face_sequence_len = min(min_face_sequence_len, sequence_len)
            for j in range(0, max(1, sequence_len - self.block_size + self.padding + 1), config.sequence_stride):  # todo: possible bug? +1 added recently
                self.indices.append((i, j))
        print('Length of', split, len(self.indices))
        print('Shortest face sequence', min_face_sequence_len)
        print('Longest face sequence', max_face_sequence_len)
        self.encoder = None
        self.pre_quant = None
        self.vq = None
        self.post_quant = None
        self.decoder = None
        self.device = None

    def set_quantizer(self, encoder, pre_quant, vq, post_quant, decoder, device):
        self.encoder = encoder.eval()
        self.pre_quant = pre_quant.eval()
        self.decoder = decoder.eval()
        self.vq = vq.eval()
        self.post_quant = post_quant.eval()
        self.device = device

    @torch.no_grad()
    def get_codes(self, vertices, faces):
        triangles, normals, areas, angles, vertices, faces = create_feature_stack(vertices, faces, self.num_tokens)
        features = np.hstack([triangles, normals, areas, angles])
        face_neighborhood = np.array(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).face_neighborhood)  # type: ignore

        encoded_x = self.encoder(
            torch.from_numpy(features).float().to(self.device),
            torch.from_numpy(face_neighborhood.T).long().to(self.device),
            torch.zeros([features.shape[0]], device=self.device).long()
        )

        encoded_x = self.pre_quant(encoded_x)
        _, all_indices, _ = self.vq(encoded_x.unsqueeze(0))
        all_indices = all_indices.squeeze(0)
        if not self.vq_is_shared:
            correction = (torch.arange(0, self.vq_depth, device=self.device) * self.vq_num_codes_per_level).reshape(1, -1)
            all_indices = all_indices + correction
        inner_face_id = torch.arange(0, self.vq_depth, device=self.device).reshape(1, -1).expand(all_indices.shape[0], -1)
        outer_face_id = torch.arange(0, all_indices.shape[0], device=self.device).reshape(-1, 1).expand(-1, self.vq_depth)

        # adjust for start, end and padding tokens
        all_indices = all_indices.reshape(-1) + 3
        inner_face_id = inner_face_id.reshape(-1) + 3
        outer_face_id = outer_face_id.reshape(-1) + 3

        # add start token and end token
        all_indices = torch.cat((
            torch.tensor([self.start_sequence_token], device=self.device),
            all_indices,
            torch.tensor([self.end_sequence_token], device=self.device)
        )).long().cpu()

        inner_face_id = torch.cat((
            torch.tensor([self.start_sequence_token], device=self.device),
            inner_face_id,
            torch.tensor([self.end_sequence_token], device=self.device)
        )).long().cpu()

        outer_face_id = torch.cat((
            torch.tensor([self.start_sequence_token], device=self.device),
            outer_face_id,
            torch.tensor([self.end_sequence_token], device=self.device)
        )).long().cpu()

        return all_indices, inner_face_id, outer_face_id

    def __getitem__(self, index: int):
        i, j = self.indices[index]
        vertices = self.cached_vertices[i]
        faces = self.cached_faces[i]
        if self.scale_augment:
            vertices = normalize_vertices(scale_vertices(vertices))
        else:
            vertices = normalize_vertices(vertices)

        soup_sequence, face_in_idx, face_out_idx = self.get_codes(vertices, faces)

        # face sequence in block format
        end_index = min(j + self.block_size, len(soup_sequence))
        x_in = soup_sequence[j:end_index]
        y_in = soup_sequence[j + 1:end_index + 1]
        fpi_in = face_in_idx[j:end_index]
        fpo_in = face_out_idx[j:end_index]

        x_pad = torch.tensor([self.pad_face_token for _ in range(0, self.block_size - len(x_in))])
        y_pad = torch.tensor([self.pad_face_token for _ in range(0, len(x_in) + len(x_pad) - len(y_in))])
        fpi_in_pad = torch.tensor([self.pad_face_token for _ in range(0, self.block_size - len(fpi_in))])
        fpo_in_pad = torch.tensor([self.pad_face_token for _ in range(0, self.block_size - len(fpo_in))])

        x = torch.cat((x_in, x_pad)).long()
        y = torch.cat((y_in, y_pad)).long()
        fpi = torch.cat((fpi_in, fpi_in_pad)).long()
        fpo = torch.from_numpy(get_shifted_sequence(torch.cat((fpo_in, fpo_in_pad)).numpy())).long()

        return {
            'name': self.names[i],
            'seq_in': x,
            'seq_out': y,
            'seq_pos_inner': fpi,
            'seq_pos_outer': fpo,
        }

    def get_completion_sequence(self, i, tokens, device=torch.device("cpu")):
        vertices = normalize_vertices(self.cached_vertices[i])
        faces = self.cached_faces[i]
        soup_sequence, face_in_idx, face_out_idx = self.get_codes(vertices, faces)
        face_out_idx = torch.from_numpy(get_shifted_sequence(face_out_idx.numpy()))
        original_fseq = soup_sequence.to(device)
        if isinstance(tokens, int):
            num_pre_tokens = tokens
        else:
            num_pre_tokens = int(len(original_fseq) * tokens)
        x = (
            soup_sequence[:num_pre_tokens].to(device)[None, ...],
            face_in_idx[:num_pre_tokens].to(device)[None, ...],
            face_out_idx[:num_pre_tokens].to(device)[None, ...],
            original_fseq[None, ...],
        )
        return x

    def get_start(self, device=torch.device("cpu")):
        i = random.choice(list(range(len(self.cached_vertices))))
        x = self.get_completion_sequence(i, 11, device)
        return x

    def __len__(self) -> int:
        return len(self.indices)

    def decode(self, sequence):
        mask = torch.isin(sequence, torch.tensor([self.start_sequence_token, self.end_sequence_token, self.pad_face_token], device=sequence.device)).logical_not()
        sequence = sequence[mask]
        sequence = sequence - 3

        sequence_len = (sequence.shape[0] // self.vq_depth) * self.vq_depth
        sequence = sequence[:sequence_len].reshape(-1, self.vq_depth)
        N = sequence.shape[0]
        E, D = self.vq_num_codes_per_level, self.vq_depth
        all_codes = self.vq.get_codes_from_indices(sequence).permute(1, 2, 0)
        encoded_x = all_codes.reshape(N, E, D).sum(-1)
        encoded_x = self.post_quant(encoded_x)
        encoded_x_conv, conv_mask = create_conv_batch(encoded_x, torch.zeros([sequence.shape[0]], device=self.device).long(), 1, self.device)
        decoded_x_conv = self.decoder(encoded_x_conv)
        decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
        coords = decoded_x.argmax(-1).detach().cpu().numpy() / self.num_tokens - 0.5
        gen_vertices, gen_faces = triangle_sequence_to_mesh(coords)
        return gen_vertices, gen_faces

    def plot_sequence_lenght_stats(self):
        sequence_lengths = []
        for i in range(len(self.cached_faces)):
            sequence_len = len(self.cached_faces[i]) * self.vq_depth + 1 + 1
            sequence_lengths.append(sequence_len)
        import matplotlib.pyplot as plt
        plt.hist(sequence_lengths, bins=32)
        plt.ylim(0, 100)
        plt.show()


class QuantizedSoupCreator(torch.nn.Module):
    vq_depth_factor = 1
    def __init__(self, config, vq_cfg):
        super().__init__()
        self.vq_cfg = vq_cfg
        self.vq_depth = self.vq_cfg.embed_levels
        self.vq_is_shared = self.vq_cfg.embed_share
        self.vq_num_codes_per_level = self.vq_cfg.n_embed
        self.vq_dim = self.vq_cfg.embed_dim
        assert config.num_tokens == self.vq_cfg.num_tokens, "Number of tokens must match"
        self.block_size = config.block_size
        self.start_sequence_token = 0
        self.end_sequence_token = 1
        self.pad_face_token = 2
        self.num_tokens = config.num_tokens - 3
        self.padding = int(config.padding * self.block_size)
        self.rq_transformer_input = False
        self.encoder, self.pre_quant, self.post_quant, self.vq = self.get_rvq_encoders(config.vq_resume)

    def get_rvq_encoders(self, resume):
        return get_rvqvae_v0_encoder_vq(self.vq_cfg, resume)

    def freeze_vq(self):
        for model in [self.encoder, self.pre_quant, self.post_quant, self.vq]:
            for param in model.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def embed(self, idx):
        assert self.vq_is_shared, "Only shared embedding is supported"
        all_codes = self.vq.codebooks[0][idx].reshape(-1, self.vq_dim)
        return all_codes

    @torch.no_grad()
    def get_indices(self, x, edge_index, batch, _faces, _num_vertices):
        encoded_x = self.encoder(x, edge_index, batch)
        encoded_x = self.pre_quant(encoded_x)
        _, all_indices, _ = self.vq(encoded_x.unsqueeze(0))
        all_indices = all_indices.squeeze(0)
        if not self.vq_is_shared:
            correction = (torch.arange(0, self.vq_depth * self.vq_depth_factor, device=x.device) * self.vq_num_codes_per_level).reshape(1, -1)
            all_indices = all_indices + correction
        return all_indices

    @torch.no_grad()
    def forward(self, x, edge_index, batch, faces, num_vertices, js, force_full_sequence=False):
        for model in [self.encoder, self.pre_quant, self.post_quant, self.vq]:
            model.eval()
        if force_full_sequence:
            assert js.shape[0] == 1, "Only single mesh supported"
        all_indices = self.get_indices(x, edge_index, batch, faces, num_vertices)
        batch_size = js.shape[0]
        sequences = []
        targets = []
        position_inners = []
        position_outers = []
        max_sequence_length_x = 0
        for k in range(batch_size):
            sequence_k = all_indices[batch == k, :]
            inner_face_id_k = torch.arange(0, self.vq_depth * self.vq_depth_factor, device=x.device).reshape(1, -1).expand(sequence_k.shape[0], -1)
            outer_face_id_k = torch.arange(0, sequence_k.shape[0], device=x.device).reshape(-1, 1).expand(-1, self.vq_depth * self.vq_depth_factor)
            sequence_k = sequence_k.reshape(-1) + 3
            inner_face_id_k = inner_face_id_k.reshape(-1) + 3
            outer_face_id_k = outer_face_id_k.reshape(-1) + 3
            # add start token and end token
            
            prefix = [torch.tensor([self.start_sequence_token], device=x.device)] 
            postfix = [torch.tensor([self.end_sequence_token], device=x.device)]

            if self.rq_transformer_input:
                prefix = prefix * self.vq_depth
                postfix = postfix * self.vq_depth

            sequence_k = torch.cat(prefix + [sequence_k] + postfix).long()

            inner_face_id_k = torch.cat(prefix + [inner_face_id_k] + postfix).long()

            outer_face_id_k = torch.cat(prefix + [outer_face_id_k] + postfix).long()

            j = js[k]
            if force_full_sequence:
                end_index = len(sequence_k)
            else:
                end_index = min(j + self.block_size, len(sequence_k))
            x_in = sequence_k[j:end_index]
            if self.rq_transformer_input:
                y_in = sequence_k[j + self.vq_depth:end_index + self.vq_depth]
            else:
                y_in = sequence_k[j + 1:end_index + 1]
            fpi_in = inner_face_id_k[j:end_index]
            fpo_in = outer_face_id_k[j:end_index].cpu()

            max_sequence_length_x = max(max_sequence_length_x, len(x_in))
            pad_len_x = self.block_size - len(x_in)
            if self.rq_transformer_input:
                pad_len_x = pad_len_x + (self.vq_depth - (len(x_in) + pad_len_x) % self.vq_depth)
            x_pad = torch.tensor([self.pad_face_token for _ in range(0, pad_len_x)], device=x.device)

            pad_len_y = len(x_in) + len(x_pad) - len(y_in)
            pad_len_fpi = self.block_size - len(fpi_in)
            pad_len_fpo = self.block_size - len(fpo_in)

            if self.rq_transformer_input:
                pad_len_fpi = pad_len_fpi + (self.vq_depth - (len(fpi_in) + pad_len_fpi) % self.vq_depth)
                pad_len_fpo = pad_len_fpo + (self.vq_depth - (len(fpo_in) + pad_len_fpo) % self.vq_depth)

            y_pad = torch.tensor([self.pad_face_token for _ in range(0, pad_len_y)], device=x.device)
            fpi_in_pad = torch.tensor([self.pad_face_token for _ in range(0, pad_len_fpi)], device=x.device)
            fpo_in_pad = torch.tensor([self.pad_face_token for _ in range(0, pad_len_fpo)])

            x = torch.cat((x_in, x_pad)).long()
            y = torch.cat((y_in, y_pad)).long()
            fpi = torch.cat((fpi_in, fpi_in_pad)).long()
            fpo = torch.from_numpy(get_shifted_sequence(torch.cat((fpo_in, fpo_in_pad)).numpy())).long().to(x.device)

            sequences.append(x)
            targets.append(y)
            position_inners.append(fpi)
            position_outers.append(fpo)

        sequences = torch.stack(sequences, dim=0)[:, :max_sequence_length_x].contiguous()
        targets = torch.stack(targets, dim=0)[:, :max_sequence_length_x].contiguous()
        position_inners = torch.stack(position_inners, dim=0)[:, :max_sequence_length_x].contiguous()
        position_outers = torch.stack(position_outers, dim=0)[:, :max_sequence_length_x].contiguous()
        return sequences, targets, position_inners, position_outers

    @torch.no_grad()
    def get_completion_sequence(self, x, edge_index, face, num_vertices, tokens):
        soup_sequence, target, face_in_idx, face_out_idx = self.forward(
            x, edge_index,
            torch.zeros([x.shape[0]], device=x.device).long(), face,
            num_vertices,
            torch.zeros([1], device=x.device).long(),
            force_full_sequence=True
        )
        soup_sequence = soup_sequence[0]
        face_in_idx = face_in_idx[0]
        face_out_idx = face_out_idx[0]
        target = target[0]
        if isinstance(tokens, int):
            num_pre_tokens = tokens
        else:
            num_pre_tokens = int(len(target) * tokens)
        x = (
            soup_sequence[:num_pre_tokens][None, ...],
            face_in_idx[:num_pre_tokens][None, ...],
            face_out_idx[:num_pre_tokens][None, ...],
            target[None, ...],
        )
        return x

    def encode_sequence(self, sequence):
        N = sequence.shape[0]
        E, D = self.vq_dim, self.vq_depth
        all_codes = self.vq.get_codes_from_indices(sequence).permute(1, 2, 0)
        encoded_x = all_codes.reshape(N, E, D).sum(-1)
        return encoded_x

    @torch.no_grad()
    def decode(self, sequence, decoder):
        mask = torch.isin(sequence, torch.tensor([self.start_sequence_token, self.end_sequence_token, self.pad_face_token], device=sequence.device)).logical_not()
        sequence = sequence[mask]
        sequence = sequence - 3
        sequence_len = (sequence.shape[0] // (self.vq_depth * self.vq_depth_factor)) * (self.vq_depth * self.vq_depth_factor)
        sequence = sequence[:sequence_len].reshape(-1, self.vq_depth)
        encoded_x = self.encode_sequence(sequence)
        encoded_x = self.post_quant(encoded_x)
        encoded_x_conv, conv_mask = create_conv_batch(encoded_x, torch.zeros([encoded_x.shape[0]], device=sequence.device).long(), 1, sequence.device)
        decoded_x_conv = decoder(encoded_x_conv)
        decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
        coords = decoded_x.argmax(-1).detach().cpu().numpy() / self.num_tokens - 0.5
        gen_vertices, gen_faces = triangle_sequence_to_mesh(coords)
        return gen_vertices, gen_faces


class QuantizedSoupTripletsCreator(QuantizedSoupCreator):
    vq_depth_factor = 3
    def __init__(self, config, vq_cfg):
        super().__init__(config, vq_cfg)

    def get_rvq_encoders(self, resume):
        return get_rvqvae_v1_encoder_vq(self.vq_cfg, resume)

    @torch.no_grad()
    def get_indices(self, x, edge_index, batch, faces, num_vertices):
        encoded_x = self.encoder(x, edge_index, batch)
        encoded_x = encoded_x.reshape(encoded_x.shape[0] * 3, 192)
        encoded_x = distribute_features(encoded_x, faces, num_vertices, x.device)
        encoded_x = self.pre_quant(encoded_x)
        _, all_indices, _ = self.vq(encoded_x.unsqueeze(0))
        all_indices = all_indices.squeeze(0).reshape(-1, self.vq_depth * 3)
        if not self.vq_is_shared:
            correction = (torch.arange(0, self.vq_depth, device=x.device) * self.vq_num_codes_per_level).reshape(1, -1)
            all_indices = all_indices + correction
        return all_indices

    def encode_sequence(self, sequence):
        N = sequence.shape[0]
        E, D = self.vq_dim, self.vq_depth
        all_codes = self.vq.get_codes_from_indices(sequence).permute(1, 2, 0)
        encoded_x = all_codes.reshape(-1, 3 * E, D).sum(-1)
        return encoded_x


def distribute_features(features, face_indices, num_vertices, device):
    assert features.shape[0] == face_indices.shape[0] * face_indices.shape[1], "Features and face indices must match in size"
    vertex_features = torch.zeros([num_vertices, features.shape[1]], device=device)
    torch_scatter.scatter_mean(features, face_indices.reshape(-1), out=vertex_features, dim=0)
    distributed_features = vertex_features[face_indices.reshape(-1), :]
    return distributed_features
