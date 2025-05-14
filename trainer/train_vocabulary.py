import omegaconf
import torch_scatter
import trimesh
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import pytorch_lightning as pl
import hydra
from lightning_utilities.core.rank_zero import rank_zero_only
from pathlib import Path
import torch
from vector_quantize_pytorch import ResidualVQ

from dataset import quantize_coordinates
from dataset.triangles import create_feature_stack_from_triangles, TriangleNodesWithFaces, TriangleNodesWithFacesDataloader
from model.encoder import GraphEncoder
from model.decoder import resnet34_decoder
from model.softargmax import softargmax
from trainer import create_trainer, step, create_conv_batch
from util.visualization import plot_vertices_and_faces, triangle_sequence_to_mesh


class TriangleTokenizationGraphConv(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        if config.only_chairs:
            self.train_dataset = TriangleNodesWithFaces(config, 'train', config.scale_augment, config.shift_augment, '03001627')
            self.interesting_categories = [('03001627', "")]
            self.val_datasets = [TriangleNodesWithFaces(config, 'val', config.scale_augment_val, config.shift_augment_val, '03001627')]
        else:
            self.train_dataset = TriangleNodesWithFaces(config, 'train', config.scale_augment, config.shift_augment, None)
            self.interesting_categories = [('02828884', '_bench'), ('02871439', '_bookshelf'), ('03001627', ""), ('03211117', '_display'), ('04379243', '_table')]
            self.val_datasets = []
            for cat, name in self.interesting_categories:
                self.val_datasets.append(TriangleNodesWithFaces(config, 'val', config.scale_augment_val, config.shift_augment_val, cat))
        self.encoder = GraphEncoder(no_max_pool=config.g_no_max_pool, aggr=config.g_aggr, graph_conv=config.graph_conv, use_point_features=config.use_point_feats, output_dim=576)
        self.decoder = resnet34_decoder(512, config.num_tokens - 2, config.ce_output)
        self.pre_quant = torch.nn.Linear(192, config.embed_dim)
        self.post_quant = torch.nn.Linear(config.embed_dim * 3, 512)
        self.vq = ResidualVQ(
            dim=self.config.embed_dim,
            codebook_size=self.config.n_embed,  # codebook size
            num_quantizers=config.embed_levels,
            commitment_weight=self.config.embed_loss_weight,  # the weight on the commitment loss
            stochastic_sample_codes=True,
            sample_codebook_temp=config.stochasticity,  # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
            shared_codebook=self.config.embed_share,
            decay=self.config.code_decay,
        )
        self.register_buffer('smoothing_weight', torch.tensor([2, 10, 200, 10, 2], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        # print('compiling model...')
        # self.model = torch.compile(model)  # requires PyTorch 2.0
        self.output_dir_image_val = Path(f'runs/{self.config.experiment}/image_val')
        self.output_dir_image_val.mkdir(exist_ok=True, parents=True)
        self.output_dir_mesh_val = Path(f'runs/{self.config.experiment}/mesh_val')
        self.output_dir_mesh_val.mkdir(exist_ok=True, parents=True)
        self.output_dir_image_train = Path(f'runs/{self.config.experiment}/image_train')
        self.output_dir_image_train.mkdir(exist_ok=True, parents=True)
        self.output_dir_mesh_train = Path(f'runs/{self.config.experiment}/mesh_train')
        self.output_dir_mesh_train.mkdir(exist_ok=True, parents=True)
        self.automatic_optimization = False
        self.distribute_features_fn = distribute_features if self.config.distribute_features else dummy_distribute
        self.visualize_groundtruth()

    def configure_optimizers(self):
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.pre_quant.parameters()) + list(self.post_quant.parameters()) + list(self.vq.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.config.lr, amsgrad=True, weight_decay=self.config.weight_decay)
        max_steps = int(self.config.max_epoch * len(self.train_dataset) / self.config.batch_size)
        print('Max Steps | First cycle:', max_steps)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, first_cycle_steps=max_steps, cycle_mult=1.0,
            max_lr=self.config.lr, min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps, gamma=1.0
        )
        return [optimizer], [scheduler]

    def create_conv_batch(self, encoded_features, batch, batch_size):
        return create_conv_batch(encoded_features, batch, batch_size, self.device)

    def training_step(self, data, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        scheduler.step()  # type: ignore
        encoded_x = self.encoder(data.x, data.edge_index, data.batch)
        encoded_x = encoded_x.reshape(encoded_x.shape[0] * 3, 192)  # 3N x 192
        encoded_x = self.distribute_features_fn(encoded_x, data.faces, data.num_vertices.sum(), self.device)
        encoded_x = self.pre_quant(encoded_x)  # 3N x 192
        encoded_x, _, commit_loss = self.vq(encoded_x.unsqueeze(0))
        encoded_x = encoded_x.squeeze(0)
        commit_loss = commit_loss.mean()
        encoded_x = encoded_x.reshape(-1, 3 * encoded_x.shape[-1])
        encoded_x = self.post_quant(encoded_x)
        encoded_x_conv, conv_mask = self.create_conv_batch(encoded_x, data.batch, self.config.batch_size)
        decoded_x_conv = self.decoder(encoded_x_conv)
        if self.config.ce_output:
            decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
            decoded_tri = softargmax(decoded_x) / (self.config.num_tokens - 3) - 0.5
            _, decoded_normals, decoded_areas, decoded_angles = create_feature_stack_from_triangles(decoded_tri.reshape(-1, 3, 3))
            if self.config.use_smoothed_loss:
                otarget = torch.nn.functional.one_hot(data.y.reshape(-1), num_classes=self.config.num_tokens - 2).float()
                otarget = otarget.unsqueeze(1)
                starget = torch.nn.functional.conv1d(otarget, self.smoothing_weight, bias=None, stride=1, padding=2, dilation=1, groups=1)
                if self.config.use_multimodal_loss:
                    starget_a = starget.reshape(-1, decoded_x.shape[-2] * decoded_x_conv.shape[-1])
                    starget_a = torch.nn.functional.normalize(starget_a, p=1.0, dim=-1, eps=1e-12).squeeze(1)
                    starget_b = torch.nn.functional.normalize(starget, p=1.0, dim=-1, eps=1e-12).squeeze(1)
                    loss = torch.nn.functional.cross_entropy(decoded_x.reshape(-1, decoded_x.shape[-2] * decoded_x_conv.shape[-1]), starget_a).mean()
                    loss = loss * 0.1 + torch.nn.functional.cross_entropy(decoded_x.reshape(-1, decoded_x.shape[-1]), starget_b).mean()
                else:
                    starget = torch.nn.functional.normalize(starget, p=1.0, dim=-1, eps=1e-12).squeeze(1)
                    loss = torch.nn.functional.cross_entropy(decoded_x.reshape(-1, decoded_x.shape[-1]), starget).mean()
            else:
                loss = torch.nn.functional.cross_entropy(decoded_x.reshape(-1, decoded_x.shape[-1]), data.y.reshape(-1), reduction='mean')
            y_coords = data.y / (self.config.num_tokens - 3) - 0.5
            loss_tri = torch.nn.functional.mse_loss(decoded_tri, y_coords, reduction='mean')
            loss_normals = torch.nn.functional.mse_loss(decoded_normals, data.x[:, 9:12], reduction='mean')
            loss_areas = torch.nn.functional.mse_loss(decoded_areas, data.x[:, 12:13], reduction='mean')
            loss_angles = torch.nn.functional.mse_loss(decoded_angles, data.x[:, 13:16], reduction='mean')
        else:
            decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :]
            _, decoded_normals, decoded_areas, decoded_angles = create_feature_stack_from_triangles(decoded_x.reshape(-1, 3, 3))
            loss = torch.nn.functional.mse_loss(decoded_x, data.y, reduction='mean')
            loss_tri = torch.nn.functional.mse_loss(decoded_x, data.y, reduction='mean')
            loss_normals = torch.nn.functional.mse_loss(decoded_normals, data.x[:, 9:12], reduction='mean')
            loss_areas = torch.nn.functional.mse_loss(decoded_areas, data.x[:, 12:13], reduction='mean')
            loss_angles = torch.nn.functional.mse_loss(decoded_angles, data.x[:, 13:16], reduction='mean')

        acc = self.get_accuracy(decoded_x, data.y)
        acc_triangle = self.get_triangle_accuracy(decoded_x, data.y)
        self.log("train/ce_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/mse_loss", loss_tri.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/norm_loss", loss_normals.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/area_loss", loss_areas.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/angle_loss", loss_angles.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/embed_loss", commit_loss.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/acc", acc.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/acc_tri", acc_triangle.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        loss = loss + loss_tri * self.config.tri_weight + loss_normals * self.config.norm_weight + loss_areas * self.config.area_weight + loss_angles * self.config.angle_weight + commit_loss
        # loss = loss + loss_tri * self.config.tri_weight + commit_loss
        loss = loss / self.config.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        self.manual_backward(loss)
        # accumulate gradients of `n` batches
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            step(optimizer, [self.encoder, self.decoder, self.pre_quant, self.post_quant])
            optimizer.zero_grad(set_to_none=True)  # type: ignore
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)  # type: ignore

    def validation_step(self, data, batch_idx, dataloader_idx):
        encoded_x = self.encoder(data.x, data.edge_index, data.batch)
        encoded_x = encoded_x.reshape(encoded_x.shape[0] * 3, 192)
        encoded_x = self.distribute_features_fn(encoded_x, data.faces, data.num_vertices.sum(), self.device)
        encoded_x = self.pre_quant(encoded_x)
        encoded_x, _, commit_loss = self.vq(encoded_x.unsqueeze(0))
        encoded_x = encoded_x.squeeze(0)
        commit_loss = commit_loss.mean()
        encoded_x = encoded_x.reshape(-1, 3 * encoded_x.shape[-1])
        encoded_x = self.post_quant(encoded_x)
        encoded_x_conv, conv_mask = self.create_conv_batch(encoded_x, data.batch, self.config.batch_size)
        decoded_x_conv = self.decoder(encoded_x_conv)
        if self.config.ce_output:
            decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
            decoded_c = softargmax(decoded_x) / (self.config.num_tokens - 3) - 0.5
            loss = torch.nn.functional.cross_entropy(decoded_x.reshape(-1, decoded_x.shape[-1]), data.y.reshape(-1), reduction='mean')
            y_coords = data.y / (self.config.num_tokens - 3) - 0.5
            loss_c = torch.nn.functional.mse_loss(decoded_c, y_coords, reduction='mean')
        else:
            decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :]
            loss = torch.nn.functional.mse_loss(decoded_x, data.y, reduction='mean')
            loss_c = torch.nn.functional.mse_loss(decoded_x, data.y, reduction='mean')
        acc = self.get_accuracy(decoded_x, data.y)
        acc_triangle = self.get_triangle_accuracy(decoded_x, data.y)
        if not torch.isnan(loss).any():
            self.log(f"val/ce_loss{self.interesting_categories[dataloader_idx][1]}", loss.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if not torch.isnan(loss_c).any():
            self.log(f"val/mse_loss{self.interesting_categories[dataloader_idx][1]}", loss_c.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if not torch.isnan(commit_loss).any():
            self.log(f"val/embed_loss{self.interesting_categories[dataloader_idx][1]}", commit_loss.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if not torch.isnan(acc).any():
            self.log(f"val/acc{self.interesting_categories[dataloader_idx][1]}", acc.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if not torch.isnan(acc_triangle).any():
            self.log(f"val/acc_tri{self.interesting_categories[dataloader_idx][1]}", acc_triangle.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    @rank_zero_only
    def on_validation_epoch_end(self):
        category_names = [""] + [x[0].strip("_") for x in self.interesting_categories]
        for didx, dataset in enumerate([self.train_dataset] + self.val_datasets):
            output_dir_image = self.output_dir_image_train if didx == 0 else self.output_dir_image_val
            output_dir_mesh = self.output_dir_mesh_train if didx == 0 else self.output_dir_mesh_val
            for k in range(self.config.num_val_samples):
                data = dataset.get(k * (len(dataset) // self.config.num_val_samples) % len(dataset))
                encoded_x = self.encoder(data.x.to(self.device), data.edge_index.to(self.device), torch.zeros([data.x.shape[0]], device=self.device).long())
                encoded_x = encoded_x.reshape(encoded_x.shape[0] * 3, 192)
                encoded_x = self.distribute_features_fn(encoded_x, data.faces.to(self.device), data.num_vertices, self.device)
                encoded_x = self.pre_quant(encoded_x)
                encoded_x, _, _ = self.vq(encoded_x.unsqueeze(0))
                encoded_x = encoded_x.squeeze(0)
                encoded_x = encoded_x.reshape(-1, 3 * encoded_x.shape[-1])
                encoded_x = self.post_quant(encoded_x)
                encoded_x_conv, conv_mask = self.create_conv_batch(encoded_x, torch.zeros([data.x.shape[0]], device=self.device).long(), 1)
                decoded_x_conv = self.decoder(encoded_x_conv)
                if self.config.ce_output:
                    decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
                    coords = decoded_x.argmax(-1).detach().cpu().numpy() / (self.config.num_tokens - 3) - 0.5
                else:
                    decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :]
                    coords = decoded_x.detach().cpu().numpy()
                gen_vertices, gen_faces = triangle_sequence_to_mesh(coords)
                plot_vertices_and_faces(gen_vertices, gen_faces, output_dir_image / f"{self.global_step:06d}_{category_names[didx]}_{k}.jpg")
                try:
                    trimesh.Trimesh(vertices=gen_vertices, faces=gen_faces, process=False).export(output_dir_mesh / f"{self.global_step:06d}_{category_names[didx]}_{k}.obj")
                except Exception as e:
                    pass  # sometimes the mesh is invalid (ngon) and we don't want to crash

    def visualize_groundtruth(self):
        category_names = [""] + [x[0].strip("_") for x in self.interesting_categories]
        for didx, dataset in enumerate([self.train_dataset] + self.val_datasets):
            output_dir_image = self.output_dir_image_train if didx == 0 else self.output_dir_image_val
            for k in range(self.config.num_val_samples):
                data = dataset.get(k * (len(dataset) // self.config.num_val_samples) % len(dataset))
                if self.config.ce_output:
                    coords = data.y / (self.config.num_tokens - 3) - 0.5
                else:
                    coords = data.y
                gen_vertices, gen_faces = triangle_sequence_to_mesh(coords)
                plot_vertices_and_faces(gen_vertices, gen_faces, output_dir_image / f"GT_{category_names[didx]}_{k}.jpg")

    def train_dataloader(self):
        return TriangleNodesWithFacesDataloader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=not self.config.overfit, num_workers=self.config.num_workers, pin_memory=True)

    def val_dataloader(self):
        dataloaders = []
        for val_dataset in self.val_datasets:
            dataloaders.append(TriangleNodesWithFacesDataloader(val_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True, num_workers=self.config.num_workers))
        return dataloaders

    def get_accuracy(self, x, y):
        if self.config.ce_output:
            return (x.argmax(-1).reshape(-1) == y.reshape(-1)).sum() / (x.shape[0] * x.shape[1])
        return (quantize_coordinates(x, self.config.num_tokens - 2).reshape(-1) == quantize_coordinates(y, self.config.num_tokens - 2).reshape(-1)).sum() / (x.shape[0] * x.shape[1])

    def get_triangle_accuracy(self, x, y):
        if self.config.ce_output:
            return torch.all(x.argmax(-1).reshape(-1, 9) == y.reshape(-1, 9), dim=-1).sum() / x.shape[0]
        return torch.all((quantize_coordinates(x, self.config.num_tokens - 2).reshape(-1, 9) == quantize_coordinates(y, self.config.num_tokens - 2).reshape(-1, 9)), dim=-1).sum() / (x.shape[0])


def distribute_features(features, face_indices, num_vertices, device):
    # N = num triangles
    # features is N3 x 192
    # face_indices is N x 3
    assert features.shape[0] == face_indices.shape[0] * face_indices.shape[1], "Features and face indices must match in size"
    vertex_features = torch.zeros([num_vertices, features.shape[1]], device=device)
    torch_scatter.scatter_mean(features, face_indices.reshape(-1), out=vertex_features, dim=0)
    distributed_features = vertex_features[face_indices.reshape(-1), :]
    return distributed_features


def dummy_distribute(features, _face_indices, _n, _device):
    return features


@hydra.main(config_path='../config', config_name='meshgpt', version_base='1.2')
def main(config):
    trainer = create_trainer("TriangleTokens", config)
    model = TriangleTokenizationGraphConv(config)
    trainer.fit(model, ckpt_path=config.resume)


if __name__ == '__main__':
    main()
