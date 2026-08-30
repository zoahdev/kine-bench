"""KINE-EMB-1 (v0.3): embodied imagination probe on MuJoCo physics simulations.

Can the predictor "imagine" the future of out-of-distribution physical dynamics
rendered by a physics engine with exact ground truth? We render three MuJoCo
scenarios (falling-and-bouncing ball, pendulum, toppling box), mask the temporal
second half, and score the predictor's imagined future against the target
encoder's real future tokens — the KINE-FUT-1 protocol applied to simulation.
Requires `pip install mujoco` (skipped gracefully when absent).
"""

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_BALL = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <camera name="fixed" pos="0 -1.6 0.6" xyaxes="1 0 0 0 0.34 0.94"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.92 1"/>
    <body pos="0 0 1.2">
      <freejoint/>
      <geom type="sphere" size="0.09" mass="0.5" rgba="0.85 0.25 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

_PENDULUM = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <camera name="fixed" pos="0 -1.4 0.3" xyaxes="1 0 0 0 0.3 0.95"/>
    <body pos="0 0 0.9">
      <joint type="hinge" axis="1 0 0" damping="0.02"/>
      <geom type="capsule" fromto="0 0 0  0 0 -0.7" size="0.02" rgba="0.6 0.65 0.7 1"/>
      <body pos="0 0 -0.7">
        <geom type="sphere" size="0.09" mass="1" rgba="0.25 0.5 0.85 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_TOPPLE = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <camera name="fixed" pos="0 -1.6 0.5" xyaxes="1 0 0 0 0.3 0.95"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.92 1"/>
    <body pos="-0.25 0 0.16">
      <freejoint/>
      <geom type="box" size="0.035 0.035 0.16" mass="0.8" rgba="0.8 0.6 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

SCENARIOS = {
    "ball_fall": (_BALL, {"qvel": None, "steps_per_frame": 30}),
    "pendulum": (_PENDULUM, {"qpos0": 1.1, "steps_per_frame": 20}),
    "box_topple": (_TOPPLE, {"qvel": [0.55, 0, 0, 0, 0, 0], "steps_per_frame": 25}),
}


def _render_scenario(xml, opts, num_frames, size):
    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    if opts.get("qpos0") is not None:
        d.qpos[0] = opts["qpos0"]
    if opts.get("qvel") is not None:
        d.qvel[:] = opts["qvel"]
    r = mujoco.Renderer(m, height=size, width=size)
    frames = []
    mujoco.mj_forward(m, d)
    for i in range(num_frames):
        if i > 0:
            for _ in range(opts["steps_per_frame"]):
                mujoco.mj_step(m, d)
        r.update_scene(d, camera="fixed")
        frames.append(r.render().copy())
    r.close()
    x = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def embodied_imagination(model, device, num_frames=16, img_size=224, seed=0):
    """KINE-EMB-1: imagined-future cosine fidelity on physics-engine videos."""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return {"cosine": None, "random_baseline": None,
                "error": "mujoco not installed (pip install mujoco)"}

    torch.manual_seed(seed)
    gt_n, gh, gw = model.grid
    pos = torch.arange(gt_n * gh * gw, device=device)
    t_of = pos // (gh * gw)
    mask_flat = t_of >= gt_n // 2
    mask_idx = pos[mask_flat].unsqueeze(0)
    vis_idx = pos[~mask_flat].unsqueeze(0)

    per, cos_list, base_list = {}, [], []
    for name, (xml, opts) in SCENARIOS.items():
        video = _render_scenario(xml, opts, num_frames, img_size).unsqueeze(0).to(device)
        full = F.normalize(model.target(video), dim=-1)
        target = torch.gather(full, 1, mask_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
        visible = model.encoder(video, visible_idx=vis_idx)
        pred = model.predictor(visible, vis_idx, mask_idx)
        cos = F.cosine_similarity(pred, target, dim=-1).mean().item()
        flat = full.reshape(-1, full.shape[-1])
        perm = torch.randperm(flat.shape[0], device=device)
        base = F.cosine_similarity(flat, flat[perm], dim=-1).mean().item()
        per[name] = {"cosine": round(float(cos), 4), "random_baseline": round(float(base), 4)}
        cos_list.append(cos)
        base_list.append(base)

    return {
        "cosine": round(float(sum(cos_list) / len(cos_list)), 4),
        "random_baseline": round(float(sum(base_list) / len(base_list)), 4),
        "n_scenarios": len(SCENARIOS),
        "per_scenario": per,
    }
