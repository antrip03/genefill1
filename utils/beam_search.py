# utils/beam_search.py
import torch
import torch.nn.functional as F

BASES = ['A', 'C', 'G', 'T']
base_to_idx = {b: i for i, b in enumerate(BASES)}
idx_to_base = {i: b for i, b in enumerate(BASES)}

def wave_beam_search(model, left_ctx, right_ctx, gap_length,
                     beam_size=5, beam_expand=10, wave_period=5, device="cuda"):
    model.eval()
    candidates = [("", 0.0)]

    for step in range(gap_length):
        cur_beam = beam_expand if (step > 0 and step % wave_period == 0) else beam_size
        new_cands = []

        for seq, seq_logp in candidates:
            if len(seq) > 0:
                gap_so_far = torch.zeros(1, 4, len(seq), device=device)
                for i, b in enumerate(seq):
                    gap_so_far[0, base_to_idx[b], i] = 1.0
            else:
                gap_so_far = torch.zeros(1, 4, 0, device=device)

            remaining = gap_length - len(seq)
            if remaining > 0:
                pad = torch.zeros(1, 4, remaining, device=device)
                gap_input = torch.cat([gap_so_far, pad], dim=2)
            else:
                gap_input = gap_so_far

            with torch.no_grad():
                logits = model(left_ctx, gap_input, right_ctx)
                logp_step = F.log_softmax(logits[0, :, step], dim=0)

            for i in range(4):
                b = idx_to_base[i]
                new_seq = seq + b
                new_logp = seq_logp + logp_step[i].item()
                new_cands.append((new_seq, new_logp))

        new_cands.sort(key=lambda x: x[1], reverse=True)
        candidates = new_cands[:cur_beam]

    best_seq, best_logp = candidates[0]
    return best_seq, best_logp
