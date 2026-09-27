"""Exact top-K inner-product search on the GPU, tiled to fit a small VRAM budget."""
import numpy as np
import torch

BLOCK = 128


def _block_topk(s: torch.Tensor, k: int):
    """Exact top-k per row via block maxima.

    Any element in a row's top-k lies in a block whose maximum is >= the k-th largest
    value, so it is inside the k blocks with the largest maxima. Scanning only those
    k*BLOCK scores replaces a full-width topk (the dominant cost) with a cheap max-pool.
    """
    n = s.shape[1]
    pad = (-n) % BLOCK
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=float("-inf"))
    blocks = s.view(s.shape[0], -1, BLOCK)
    kb = min(k, blocks.shape[1])
    _, top_blocks = torch.topk(blocks.amax(2), kb, dim=1)                      # (Q, kb)
    cand = torch.gather(blocks, 1, top_blocks.unsqueeze(2).expand(-1, -1, BLOCK))  # (Q, kb, BLOCK)
    cand_idx = (top_blocks.unsqueeze(2) * BLOCK + torch.arange(BLOCK, device=s.device)).reshape(s.shape[0], -1)
    ts, pos = torch.topk(cand.reshape(s.shape[0], -1), min(k, n), dim=1)
    return ts, torch.gather(cand_idx, 1, pos)


def topk_search(index: np.ndarray, queries: np.ndarray, k: int,
                q_chunk: int = 4096, i_chunk: int = 262144, device: str = "cuda"):
    """Return (scores, idx) arrays of shape (len(queries), k), best first.

    ``index`` stays resident on the GPU in fp16; queries stream through in chunks and
    the index is scanned in tiles, merging a running top-k per query.
    """
    k = min(k, len(index))
    idx_gpu = torch.from_numpy(index).to(device, torch.float16)
    out_s = np.empty((len(queries), k), dtype=np.float16)
    out_i = np.empty((len(queries), k), dtype=np.int32)
    with torch.inference_mode():
        for q0 in range(0, len(queries), q_chunk):
            q = torch.from_numpy(queries[q0:q0 + q_chunk]).to(device, torch.float16)
            best_s = best_i = None
            for i0 in range(0, len(index), i_chunk):
                s = q @ idx_gpu[i0:i0 + i_chunk].T
                ts, ti = _block_topk(s, min(k, s.shape[1]))
                ti = ti.int() + i0
                if best_s is None:
                    best_s, best_i = ts, ti
                else:
                    cs, ci = torch.cat([best_s, ts], 1), torch.cat([best_i, ti], 1)
                    best_s, pos = torch.topk(cs, k, dim=1)
                    best_i = torch.gather(ci, 1, pos)
            out_s[q0:q0 + len(q)] = best_s.cpu().numpy()
            out_i[q0:q0 + len(q)] = best_i.cpu().numpy()
    del idx_gpu
    torch.cuda.empty_cache()
    return out_s, out_i
