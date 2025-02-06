import os
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity

from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from utils.convert import pano_to_lidar
from utils.sim3.sim3 import align_ate_c2b_use_a2b, compute_ate

def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean
    # distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2

class RaydropMeter:
    def __init__(self, ratio):
        self.V = []
        self.N = 0
        self.ratio = ratio

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        results = []

        rmse = (truths - preds) ** 2
        rmse = np.sqrt(rmse.mean())
        results.append(rmse)

        # raydrop = np.where(preds > self.ratio, 1, 0)
        # acc = (raydrop==truths).sum()/raydrop.size
        # results.append(acc)

        for i in range(9):
            ratio = 0.1*(i+1)
            raydrop = np.where(preds > ratio, 1, 0)
            acc = (raydrop==truths).sum()/raydrop.size
            results.append(acc)

        self.V.append(results)
        self.N += 1

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(os.path.join(prefix, "raydrop error"), self.measure()[0], global_step)

    def report(self):
        return f"Raydrop_error (rmse, accuracy) = {self.measure()}"
class DepthMeter:
    def __init__(self, scale, lpips_fn=None):
        self.V = []
        self.N = 0
        self.scale = scale
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips_fn = lpips.LPIPS(net='alex').eval()

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        depth_error = self.compute_depth_errors(truths, preds)

        depth_error = list(depth_error)
        self.V.append(depth_error)
        self.N += 1

    def compute_depth_errors(
        self, gt, pred, min_depth=1e-6, max_depth=80, thresh_set=1.25
    ):  
        pred[pred < min_depth] = min_depth
        pred[pred > max_depth] = max_depth
        gt[gt < min_depth] = min_depth
        gt[gt > max_depth] = max_depth
        
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < thresh_set).mean()
        a2 = (thresh < thresh_set**2).mean()
        a3 = (thresh < thresh_set**3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        medae =  np.median(np.abs(gt - pred))

        lpips_loss = self.lpips_fn(torch.from_numpy(pred).squeeze(0), 
                                   torch.from_numpy(gt).squeeze(0), normalize=True).item()

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )

        psnr = 10 * np.log10(max_depth**2 / np.mean((pred - gt) ** 2))

        return rmse, medae, a1, a2, a3, lpips_loss, ssim, psnr

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(
            os.path.join(prefix, f"depth error{suffix}"), self.measure()[0], global_step
        )

    def report(self):
        return f"Depth_error = {self.measure()}"
class IntensityMeter:
    def __init__(self, scale, lpips_fn=None):
        self.V = []
        self.N = 0
        self.scale = scale
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips_fn = lpips.LPIPS(net='alex').eval()

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        intensity_error = self.compute_intensity_errors(truths, preds)

        intensity_error = list(intensity_error)
        self.V.append(intensity_error)
        self.N += 1

    def compute_intensity_errors(
        self, gt, pred, min_intensity=1e-6, max_intensity=1.0, thresh_set=1.25
    ):
        pred[pred < min_intensity] = min_intensity
        pred[pred > max_intensity] = max_intensity
        gt[gt < min_intensity] = min_intensity
        gt[gt > max_intensity] = max_intensity

        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < thresh_set).mean()
        a2 = (thresh < thresh_set**2).mean()
        a3 = (thresh < thresh_set**3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        medae =  np.median(np.abs(gt - pred))

        lpips_loss = self.lpips_fn(torch.from_numpy(pred).squeeze(0), 
                                   torch.from_numpy(gt).squeeze(0), normalize=True).item()

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )

        psnr = 10 * np.log10(max_intensity**2 / np.mean((pred - gt) ** 2))

        return rmse, medae, a1, a2, a3, lpips_loss, ssim, psnr

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(
            os.path.join(prefix, f"intensity error{suffix}"), self.measure()[0], global_step
        )

    def report(self):
        return f"Inten_error = {self.measure()}"
class PointsMeter:
    def __init__(self, scale, intrinsics):
        self.V = []
        self.N = 0
        self.scale = scale
        self.intrinsics = intrinsics

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        chamLoss = chamfer_3DDist()
        pred_lidar = pano_to_lidar(preds[0], self.intrinsics)
        gt_lidar = pano_to_lidar(truths[0], self.intrinsics)

        dist1, dist2, idx1, idx2 = chamLoss(
            torch.FloatTensor(pred_lidar[None, ...]).cuda(),
            torch.FloatTensor(gt_lidar[None, ...]).cuda(),
        )
        chamfer_dis = dist1.mean() + dist2.mean()
        threshold = 0.05  # monoSDF
        f_score, precision, recall = fscore(dist1, dist2, threshold)
        f_score = f_score.cpu()[0]

        self.V.append([chamfer_dis.cpu(), f_score])

        self.N += 1

    def measure(self):
        # return self.V / self.N
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(os.path.join(prefix, "CD"), self.measure()[0], global_step)

    def report(self):
        return f"Point_error (CD, f-score) = {self.measure()}"

class TrajectoryMeter:
    def __init__(self, scale,offset):
        self.scale=scale
        self.offset=offset
    def prepare_inputs(self,poses):
        poses[:, :3, -1] = poses[:, :3, -1]/self.scale + self.offset
        return poses
    def update(self,preds,gt):
        preds=self.prepare_inputs(preds)
        gt=self.prepare_inputs(gt)
        res,s,R,t=align_ate_c2b_use_a2b(preds,gt,konwn_scale=True)
        ate, rpe_trans, rpe_rot = compute_ate(res,gt)
        self.ate=ate
        self.rpe_trans=rpe_trans
        self.rpe_rot=rpe_rot
    def measure(self):
        return self.ate,self.rpe_trans,self.rpe_rot
    def write(self,writer,global_step, prefix="",suffix=""):
        writer.add_scalar(os.path.join(prefix, "trajectory"), self.measure()[0], global_step)
    def report(self):
        return f"Trajectory_error (ATE, RPE_trans, RPE_rot) = {self.measure()}"
    def clear(self):
        pass
    
    