"""Anchored Set Aggregation (ASA): a set encoder over the members of each rotated-WBF cluster.

score  s = sigmoid(logit p0 + delta(h)), p0 the original WBF score, delta zero-initialised and
         penalised, so that the untrained model is WBF;
box    b = sum_i alpha_i * aligned member i, alpha = softmax_i(log c_i + gamma([e_i, h])), gamma's
         last layer zero-initialised, so that the untrained box is the WBF box.
h pools the member embeddings e_i = phi(x_i) with the cluster descriptor, a class embedding and
logit p0. Training data are the clusters of the detector's own training-split predictions;
loss = BCE(s, IoU >= 0.5) + lambda_delta * delta^2 + lambda_b * log(1 + KL(fused || ground truth))
with boxes as 2-D Gaussians. Geometry is computed in normalised coordinates with y scaled by the
image's height / width, so boxes are rectangles there; IoU is invariant to this affine map.
"""
import math

import numpy as np

from .clusters import CLASS_COL, M_MAX

# the settings of the paper (Section 4.6), used for every detector, dataset and family
DEFAULTS = dict(epochs=4, batch=2048, lr=2e-3, box_weight=1.0, delta_l2=0.05, box_mode="corner", seed=0)


def wbf_avg_score(X, n_views):
    """Original WBF score p0: mean member confidence x min(#boxes, #views) / #views."""
    return X[:, 4] * np.minimum(X[:, 2], n_views) / n_views


# ---------------------------------------------------------------------------- geometry (torch)

def poly_to_gauss(poly, aspect):
    """poly [..., 8] normalised, aspect [...] -> mean [..., 2], covariance [..., 2, 2] in aspect space."""
    import torch
    p = poly.reshape(*poly.shape[:-1], 4, 2).clone()
    p[..., 1] = p[..., 1] * aspect[..., None]
    mu = p.mean(-2)
    e1, e2 = p[..., 1, :] - p[..., 0, :], p[..., 3, :] - p[..., 0, :]
    sig = (e1[..., :, None] * e1[..., None, :] + e2[..., :, None] * e2[..., None, :]) / 12.0
    return mu, sig + 1e-10 * torch.eye(2, dtype=sig.dtype)


def gauss_to_poly(mu, sig, aspect):
    import torch
    lam, vec = torch.linalg.eigh(sig)
    half = torch.sqrt(3.0 * lam.clamp_min(1e-12))
    a, b = vec[..., :, 0] * half[..., 0:1], vec[..., :, 1] * half[..., 1:2]
    pts = torch.stack([mu + a + b, mu - a + b, mu - a - b, mu + a - b], -2)
    pts[..., 1] = pts[..., 1] / aspect[..., None]
    return pts.reshape(*pts.shape[:-2], 8)


def kld(mu_p, sig_p, mu_q, sig_q):
    """KL(N_p || N_q) for 2-D Gaussians."""
    import torch
    inv_q = torch.linalg.inv(sig_q)
    d = (mu_q - mu_p)[..., :, None]
    tr = (inv_q @ sig_p).diagonal(dim1=-2, dim2=-1).sum(-1)
    maha = (d.transpose(-1, -2) @ inv_q @ d)[..., 0, 0]
    return 0.5 * (tr + maha - 2.0 + torch.logdet(sig_q) - torch.logdet(sig_p))


def align_to_ref(pts):
    """pts [n, M, 4, 2]: reorder each member's vertices to best match member 0 (as WBF does)."""
    import torch
    cands = []
    for q in (pts, pts.flip(-2)):
        for k in range(4):
            cands.append(torch.roll(q, -k, dims=-2))
    C = torch.stack(cands, 2)                                  # [n, M, 8, 4, 2]
    cost = ((C - pts[:, :1, None]) ** 2).sum((-1, -2))         # [n, M, 8]
    best = cost.argmin(2)
    return torch.gather(C, 2, best[:, :, None, None, None].expand(-1, -1, 1, 4, 2))[:, :, 0]


def corner_gauss(pts):
    """Gaussian of the parallelogram spanned by vertices 0, 1, 3 of pts [n, 4, 2] (aspect space)."""
    import torch
    mu = pts.mean(-2)
    e1, e2 = pts[..., 1, :] - pts[..., 0, :], pts[..., 3, :] - pts[..., 0, :]
    sig = (e1[..., :, None] * e1[..., None, :] + e2[..., :, None] * e2[..., None, :]) / 12.0
    return mu, sig + 1e-10 * torch.eye(2, dtype=sig.dtype)


def fused_gauss(box):
    return corner_gauss(box) if not isinstance(box, tuple) else box


def fused_poly(box, asp):
    if isinstance(box, tuple):
        return gauss_to_poly(box[0], box[1], asp)
    pts = box.clone()
    pts[..., 1] = pts[..., 1] / asp[:, None]
    return pts.reshape(len(pts), 8)


# ---------------------------------------------------------------------------- tensors

CLUSTER_KEYS = ("X", "y", "fpoly", "gpoly", "cls", "name", "aspect", "mcount")
MEMBER_KEYS = ("mview", "mconf", "mpoly", "miou")


def _slice_clusters(z, a, b):
    off = np.concatenate([[0], np.cumsum(z["mcount"])])
    out = {k: z[k][a:b] for k in CLUSTER_KEYS}
    out.update({k: z[k][off[a]:off[b]] for k in MEMBER_KEYS})
    return out


def to_tensors(z, n_views, chunk=200_000):
    """Padded member tensors and cluster tensors from a ``build_members`` cache.

    Computed in chunks of clusters: the vertex alignment materialises eight orderings of every
    member, and freed CPU memory is not always returned to the operating system."""
    import torch
    n = len(z["y"])
    if n <= chunk:
        return _to_tensors(z, n_views)
    parts = [_to_tensors(_slice_clusters(z, a, min(a + chunk, n)), n_views) for a in range(0, n, chunk)]
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def _to_tensors(z, n_views):
    import torch
    n = len(z["y"])
    cnt = z["mcount"]
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    idx = start[:, None] + np.arange(M_MAX)[None, :]
    mask = np.arange(M_MAX)[None, :] < cnt[:, None]
    idx = np.where(mask, idx, 0)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    asp = t(z["aspect"]).float()
    mpoly = t(z["mpoly"][idx]).float()                                    # [n, M, 8]
    mmu, msig = poly_to_gauss(mpoly, asp[:, None].expand(n, M_MAX))
    fmu, fsig = poly_to_gauss(t(z["fpoly"]).float(), asp)
    # member geometry relative to the WBF box, whitened by the WBF box's Gaussian
    lam, vec = torch.linalg.eigh(fsig)
    w_half = vec @ torch.diag_embed(lam.clamp_min(1e-12).rsqrt()) @ vec.transpose(-1, -2)
    dmu = ((mmu - fmu[:, None])[..., None, :] @ w_half[:, None]).squeeze(-2)
    wsig = w_half[:, None] @ msig @ w_half[:, None]
    conf = t(z["mconf"][idx]).float().clamp(1e-6, 1.0)
    mfeat = torch.cat([
        torch.nn.functional.one_hot(t(z["mview"][idx]), n_views).float(),
        conf[..., None], conf.log()[..., None], t(z["miou"][idx]).float()[..., None],
        dmu.clamp(-5, 5), (wsig[..., 0, 0] - 1).clamp(-5, 5)[..., None],
        wsig[..., 0, 1].clamp(-5, 5)[..., None], (wsig[..., 1, 1] - 1).clamp(-5, 5)[..., None],
        (torch.arange(M_MAX).float() / M_MAX).expand(n, M_MAX)[..., None],
    ], -1)
    X = z["X"]
    xc = np.delete(X, CLASS_COL, axis=1).astype(np.float32)
    xc[:, 12] = np.clip(xc[:, 12], -20, 0)          # log area
    xc[:, 13] = np.log(np.clip(xc[:, 13], 1, 100))  # log aspect ratio
    prior = np.clip(wbf_avg_score(X, n_views), 1e-6, 1 - 1e-6)
    mpts = mpoly.reshape(n, M_MAX, 4, 2).clone()
    mpts[..., 1] = mpts[..., 1] * asp[:, None, None]
    mpts = align_to_ref(mpts)
    gmu, gsig = poly_to_gauss(t(z["gpoly"]).float(), asp)
    return dict(mfeat=mfeat, mask=t(mask), conf=conf, mmu=mmu, msig=msig, xc=t(xc),
                cls=t(z["cls"]), prior=t(prior).float(), y=t(z["y"]).float(),
                gmu=gmu, gsig=gsig, asp=asp, mpts=mpts)


# ---------------------------------------------------------------------------- model

def make_net(d_member, d_cluster, n_classes, hidden=64, box_mode="corner", anchor=True, members=True):
    """anchor=False predicts the score freely; members=False zeroes the member descriptors
    (ablations of Table 9)."""
    import torch
    from torch import nn

    class SetAggregator(nn.Module):
        def __init__(self):
            super().__init__()
            self.member = nn.Sequential(nn.Linear(d_member, hidden), nn.ReLU(),
                                        nn.Linear(hidden, hidden), nn.ReLU())
            self.cls_emb = nn.Embedding(n_classes, 16)
            self.cluster = nn.Sequential(nn.Linear(2 * hidden + d_cluster + 16 + 1, 2 * hidden),
                                         nn.ReLU(), nn.Linear(2 * hidden, hidden), nn.ReLU())
            self.delta = nn.Linear(hidden, 1)
            self.weight = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            for lin in (self.delta, self.weight[-1]):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            self.register_buffer("xm", torch.zeros(d_cluster))
            self.register_buffer("xs", torch.ones(d_cluster))

        def forward(self, b):
            m = self.member(b["mfeat"] if members else torch.zeros_like(b["mfeat"]))
            mask = b["mask"][..., None]
            mean = (m * mask).sum(1) / mask.sum(1).clamp_min(1)
            mx = m.masked_fill(~mask, -1e4).max(1).values
            logit_prior = torch.logit(b["prior"])
            xc = (b["xc"] - self.xm) / self.xs
            h = self.cluster(torch.cat([mean, mx, xc, self.cls_emb(b["cls"]), logit_prior[:, None]], -1))
            logit = (logit_prior if anchor else 0.0) + self.delta(h)[:, 0]
            g = self.weight(torch.cat([m, h[:, None].expand_as(m)], -1))[..., 0]
            w = torch.softmax((b["conf"].log() + g).masked_fill(~b["mask"], -1e4), 1)
            delta = logit - logit_prior
            if box_mode == "corner":
                return logit, (w[..., None, None] * b["mpts"]).sum(1), delta
            mu = (w[..., None] * b["mmu"]).sum(1)
            sig = (w[..., None, None] * b["msig"]).sum(1)
            return logit, (mu, sig), delta

    return SetAggregator()


def fit(T, n_views, n_classes, epochs=4, batch=2048, lr=2e-3, box_weight=1.0, seed=0, log=print,
        box_mode="corner", delta_l2=0.05, anchor=True, members=True):
    import torch
    torch.manual_seed(seed)
    net = make_net(T["mfeat"].shape[-1], T["xc"].shape[-1], n_classes, box_mode=box_mode,
                   anchor=anchor, members=members)
    net.xm.copy_(T["xc"].mean(0))
    net.xs.copy_(T["xc"].std(0).clamp_min(1e-3))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(T["y"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * math.ceil(n / batch), pct_start=0.1)
    bce = torch.nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = [0.0, 0.0]
        for i in range(0, n, batch):
            ix = perm[i:i + batch]
            b = {k: v[ix] for k, v in T.items()}
            logit, box, delta = net(b)
            mu, sig = fused_gauss(box)
            pos = b["y"] >= 0.5
            loss_s = bce(logit, pos.float()) + delta_l2 * (delta ** 2).mean()
            pb = pos & (torch.linalg.det(b["gsig"]) > 1e-14)  # degenerate ground truth stays out of the box loss
            loss_b = torch.log1p(kld(mu[pb], sig[pb], b["gmu"][pb], b["gsig"][pb]).clamp_min(0)).mean() \
                if pb.any() else logit.sum() * 0
            loss = loss_s + box_weight * loss_b
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot[0] += loss_s.item() * len(ix)
            tot[1] += loss_b.item() * len(ix)
        log(f"  epoch {ep + 1}/{epochs} score loss {tot[0] / n:.4f} box loss {tot[1] / n:.4f}")
    return net


def predict(net, T, batch=8192):
    """Scores and fused polygons (normalised image coordinates) for every cluster."""
    import torch
    net.eval()
    out_s, out_p = [], []
    with torch.no_grad():
        for i in range(0, len(T["y"]), batch):
            b = {k: v[i:i + batch] for k, v in T.items()}
            logit, box, _ = net(b)
            out_s.append(torch.sigmoid(logit))
            out_p.append(fused_poly(box, b["asp"]))
    return torch.cat(out_s).numpy(), torch.cat(out_p).numpy()


def save(net, path, **meta):
    """Weights and the settings ``load`` needs to rebuild the network (family, views, sizes)."""
    import torch
    torch.save({"state_dict": net.state_dict(), "meta": meta}, path)


def load(path):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    m = ckpt["meta"]
    net = make_net(m["d_member"], m["d_cluster"], m["num_classes"], box_mode=m["box_mode"])
    net.load_state_dict(ckpt["state_dict"])
    return net, m


def gauss_average_polys(T):
    """Confidence-weighted average of the members' Gaussians, without learning (ablation)."""
    import torch
    w = torch.softmax(T["conf"].log().masked_fill(~T["mask"], -1e4), 1)
    mu = (w[..., None] * T["mmu"]).sum(1)
    sig = (w[..., None, None] * T["msig"]).sum(1)
    return gauss_to_poly(mu, sig, T["asp"]).numpy()
