from pathlib import Path

p = Path("scripts/final_eval.py")
s = p.read_text(encoding="utf-8")


def sub(old, new):
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor not found exactly once: {old[:70]!r}")
    s = s.replace(old, new)


sub('ap.add_argument("--boot", type=int, default=1000)\n',
    'ap.add_argument("--boot", type=int, default=1000)\n'
    'ap.add_argument("--K", type=int, default=100, help="reranking candidates")\n'
    'ap.add_argument("--no_rm", action="store_true", help="skip reward-model scoring (model-free rewards)")\n')
sub("cands, logp = policy_topk(args.base, 100)", "cands, logp = policy_topk(args.base, args.K)")
sub("def score(cands):\n    return (",
    "def score(cands):\n    if args.no_rm:\n        z = np.zeros(cands.shape + (4,), np.float32)\n        return z, z\n    return (")
p.write_text(s, encoding="utf-8")
print("patched final_eval.py")