from pathlib import Path

p = Path("scripts/final_eval.py")
s = p.read_text(encoding="utf-8")


def sub(old, new):
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor not found exactly once: {old[:70]!r}")
    s = s.replace(old, new)


sub('choices=["ratio", "novelty"]', 'choices=["ratio", "novelty", "itemmean"]')
sub("len_sec, _ = token_stats(ev, V)\n", 'len_sec, _ = token_stats(ev, V)\n_, ITEM_MEAN = token_stats(ev, V, "itemmean")\n')
sub('    return np.log10(tokens - 1.0) if args.reward == "novelty" else P[..., 0]',
    '    if args.reward == "novelty":\n        return np.log10(tokens - 1.0)\n    if args.reward == "itemmean":\n        return ITEM_MEAN[tokens]\n    return P[..., 0]')
p.write_text(s, encoding="utf-8")
print("patched final_eval.py")