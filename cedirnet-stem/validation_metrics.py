"""Thresholded one-to-one nanoparticle validation in network pixel units."""
import numpy as np
from scipy.optimize import linear_sum_assignment


class ParticleMetrics:
    def __init__(self, distance=20):
        self.distance = distance
        self.tp = self.predicted = self.expected = 0
        self.localization = self.radius = 0.0

    def update(self, centers, radii, ground_truth, gt_radii):
        centers = np.asarray(centers).reshape(-1,2)
        ground_truth = np.asarray(ground_truth).reshape(-1,2)
        self.predicted += len(centers); self.expected += len(ground_truth)
        if not len(centers) or not len(ground_truth):
            return
        distances = np.linalg.norm(centers[:,None]-ground_truth[None,:],axis=-1)
        # Penalize forbidden edges enough to maximize valid match cardinality.
        cost = np.where(distances<=self.distance,distances,(min(distances.shape)+1)*(self.distance+1))
        rows,cols = linear_sum_assignment(cost)
        valid = distances[rows,cols]<=self.distance
        rows,cols = rows[valid],cols[valid]
        self.tp += len(rows)
        self.localization += distances[rows,cols].sum()
        self.radius += np.abs(np.asarray(radii).reshape(-1)[rows]-np.asarray(gt_radii).reshape(-1)[cols]).sum()

    def compute(self):
        p = self.tp/max(1,self.predicted); r = self.tp/max(1,self.expected)
        return dict(particle_precision=p,particle_recall=r,particle_f1=2*p*r/(p+r) if p+r else 0,
            particle_matches=self.tp,particle_localization_mae_px=self.localization/max(1,self.tp),
            particle_radius_mae_px=self.radius/max(1,self.tp))
