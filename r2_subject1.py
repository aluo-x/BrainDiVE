#!/usr/bin/env python3
"""
BrainDiVE voxel-wise R² for Subject 1, with paper-style cortical point maps.

This script preserves the NSD shared1000 train/test split from
view_r2_train_test_nsd.py and adds a bilateral 3D voxel-cloud visualization.
The 3D view colors the selected mask voxels by R². It is a voxel-space
visualization, not a FreeSurfer cortical-surface rendering.
"""
import os
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from data_maker.options import Options


def norm_id(value):
    s = str(value).strip()
    try:
        if s == "" or s.lower() == "nan":
            return None
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return s


def truthy_shared(series):
    if getattr(series.dtype, "kind", None) == "b":
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "1.0", "yes", "y"}
    )


def locate_csv(nsd_root, explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"NSD metadata CSV not found: {path}")
        return path
    candidates = sorted(Path(nsd_root).rglob("nsd_stim_info_merged.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find nsd_stim_info_merged.csv under {nsd_root}; "
            "pass --nsd-csv explicitly."
        )
    return candidates[0]


def get_shared_ids(nsd_root, csv_path, expdesign_path=None):
    df = pd.read_csv(csv_path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "cocoid" not in cols:
        raise ValueError(f"No cocoId column in {csv_path}; columns={df.columns.tolist()}")
    if "shared1000" in cols:
        shared_rows = df.loc[truthy_shared(df[cols["shared1000"]])]
        method = f"{cols['shared1000']} column"
    else:
        if expdesign_path:
            mat_path = Path(expdesign_path)
            if not mat_path.exists():
                raise FileNotFoundError(mat_path)
        else:
            candidates = sorted(Path(nsd_root).rglob("nsd_expdesign.mat"))
            if not candidates:
                raise FileNotFoundError(
                    "CSV has no shared1000 column and nsd_expdesign.mat was not found; "
                    "pass --expdesign."
                )
            mat_path = candidates[0]
        mat = loadmat(str(mat_path))
        if "sharedix" not in mat:
            raise ValueError(f"No sharedix array in {mat_path}")
        idx = np.asarray(mat["sharedix"]).reshape(-1).astype(np.int64) - 1
        if np.any(idx < 0) or np.any(idx >= len(df)):
            raise ValueError("sharedix indices out of bounds for this metadata CSV.")
        shared_rows = df.iloc[idx]
        method = f"sharedix in {mat_path}"
    ids = {norm_id(x) for x in shared_rows[cols["cocoid"]].tolist()}
    ids.discard(None)
    return ids, method, len(shared_rows)


def calc_r2(y, pred):
    rss = np.sum((y - pred) ** 2, axis=0)
    tss = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    out = np.full(y.shape[1], np.nan, dtype=np.float32)
    good = tss > 1e-12
    out[good] = 1.0 - rss[good] / tss[good]
    return out


def save_hist(r2, split, n, outdir):
    finite = r2[np.isfinite(r2)]
    path = os.path.join(outdir, f"voxelwise_r2_histogram_nsd_{split}_{n}items.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(finite, bins=80)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("R² per voxel")
    ax.set_ylabel("Number of voxels")
    ax.set_title(f"Subject 1 {split} voxel-wise R² ({n} images)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_voxel_cloud(r2, mask_path, split, n, outdir, vmax=0.5):
    if not mask_path:
        print("No --mask supplied; skipping spatial voxel map.")
        return
    with open(mask_path, "rb") as f:
        loaded_mask = pickle.load(f)

    # The shared mask file may store masks with a leading subject dimension.
    # Subject 1 is index 0 in zero-based Python indexing. If the file already
    # contains one 3D volume, use it unchanged.
    if isinstance(loaded_mask, (list, tuple)):
        if len(loaded_mask) == 1:
            loaded_mask = loaded_mask[0]
        elif len(loaded_mask) > 1 and np.asarray(loaded_mask[0]).ndim == 3:
            loaded_mask = loaded_mask[0]

    mask = np.asarray(loaded_mask).squeeze()
    if mask.ndim == 4:
        mask = mask[0]  # subject index 0 (Subject 1)
    if mask.ndim != 3:
        raise ValueError(
            f"Expected a 3D mask, or a 4D subject-first mask; got {mask.shape}"
        )
    mask = mask.astype(bool)
    coords = np.argwhere(mask)
    if len(coords) != len(r2):
        raise ValueError(
            f"Mask has {len(coords)} voxels but R² has {len(r2)} scores. "
            "Use the mask whose flattened True-voxel order matches the model outputs."
        )

    volume = np.full(mask.shape, np.nan, dtype=np.float32)
    volume[mask] = r2
    volpath = os.path.join(outdir, f"voxelwise_r2_volume_nsd_{split}_{n}items.npy")
    np.save(volpath, volume)

    # Plot the actual selected voxels in mask-grid coordinates, colored by R².
    # Three views make the spatial pattern easier to inspect without MRI software.
    views = [
        (0, 1, 2, "Axial-like view (X–Y)"),
        (0, 2, 1, "Coronal-like view (X–Z)"),
        (1, 2, 0, "Sagittal-like view (Y–Z)"),
    ]
    fig = plt.figure(figsize=(15, 5), facecolor="white")
    norm = plt.Normalize(vmin=-1.0, vmax=vmax)
    cmap = plt.get_cmap("inferno")
    for panel, (a, b, depth, title) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, panel)
        valid = np.isfinite(r2)
        sc = ax.scatter(
            coords[valid, a], coords[valid, b],
            c=r2[valid], cmap=cmap, norm=norm,
            s=3, marker="s", linewidths=0, rasterized=True
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(sc, ax=fig.axes, fraction=0.025, pad=0.02)
    cbar.set_label("Test-set R²" if split == "test" else "R²")
    fig.suptitle(f"Subject 1 voxel-wise R² — {split} set ({n} images)", y=1.02)
    fig.tight_layout()
    pngpath = os.path.join(outdir, f"voxelwise_r2_spatial_nsd_{split}_{n}items.png")
    fig.savefig(pngpath, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved spatial voxel plot: {pngpath}")
    print(f"Saved R² volume: {volpath}")


def main():
    config = Options()
    p = config.parser
    p.description = "Subject 1 voxel-wise R² and spatial plots."
    p.add_argument("--h5", default=None)
    p.add_argument("--nsd-csv", default=None)
    p.add_argument("--expdesign", default=None)
    p.add_argument("--mask", default=None, help="shared volume mask pickle; if subject-first 4D/list, uses index 0 (Subject 1)")
    p.add_argument("--subject", default="1", help="Label only; this script expects the Subject 1 HDF5 by default")
    p.add_argument("--map-vmax", type=float, default=0.5, help="Upper R² color limit (default 0.5)")
    args = config.parse()

    h5_path = Path(args.h5 or os.path.join(args.save_loc, 'eval', 'subject_1_neurips_split_VIT_last_fully_linear.h5py'))
    if not h5_path.exists():
        raise FileNotFoundError(f"Evaluation HDF5 not found: {h5_path}")
    outdir = os.path.join(args.save_loc, 'eval')
    os.makedirs(outdir, exist_ok=True)

    csv_path = locate_csv(args.nsd_root, args.nsd_csv)
    shared_ids, method, shared_rows = get_shared_ids(args.nsd_root, csv_path, args.expdesign)
    print(f"Metadata CSV: {csv_path}")
    print(f"Shared set source: {method}; rows={shared_rows}; unique COCO IDs={len(shared_ids)}")

    with h5py.File(h5_path, "r") as f:
        pairs = {}
        for key in f.keys():
            if not key.endswith("_gt"):
                continue
            sid = norm_id(key[:-3])
            predkey = key[:-3] + "_eval"
            if sid is not None and predkey in f:
                if sid in pairs:
                    raise RuntimeError(f"Duplicate normalized ID {sid}")
                pairs[sid] = (key, predkey)

        ids_found = set(pairs)
        test_ids = ids_found & shared_ids
        train_ids = ids_found - shared_ids
        print(f"Matched HDF5 stimulus pairs: {len(ids_found)}")
        print(f"Shared test IDs found: {len(test_ids)}; missing shared IDs: {len(shared_ids - ids_found)}")
        print(f"Remaining IDs treated as train: {len(train_ids)}")

        def load(ids):
            ordered = sorted(ids, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
            if not ordered:
                raise RuntimeError("No matched HDF5 items in this split.")
            gt = np.stack([f[pairs[s][0]][:].reshape(-1) for s in ordered])
            pred = np.stack([f[pairs[s][1]][:].reshape(-1) for s in ordered])
            if gt.shape != pred.shape:
                raise ValueError(f"Ground truth {gt.shape} != prediction {pred.shape}")
            return gt, pred

        y_train, p_train = load(train_ids)
        y_test, p_test = load(test_ids)

    for split, y, pred in (("train", y_train, p_train), ("test", y_test, p_test)):
        r2 = calc_r2(y, pred)
        r2path = os.path.join(outdir, f"voxelwise_r2_nsd_{split}_{len(y)}items.npy")
        np.save(r2path, r2)
        finite = r2[np.isfinite(r2)]
        print(f"\n{split.upper()}: items={len(y)}, voxels={len(r2)}, "
              f"mean R²={np.mean(finite):.4f}, median R²={np.median(finite):.4f}, "
              f"positive={100*np.mean(finite > 0):.1f}%")
        print(f"Saved voxel scores: {r2path}")
        print(f"Saved histogram: {save_hist(r2, split, len(y), outdir)}")
        save_voxel_cloud(r2, args.mask, split, len(y), outdir, args.map_vmax)

    print("\nDone.")


if __name__ == "__main__":
    main()

