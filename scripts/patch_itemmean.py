from pathlib import Path

p = Path("src/rl_lib.py")
s = p.read_text(encoding="utf-8")
if "# --- itemmean reward ---" in s:
    raise SystemExit("already patched")
s += '''

# --- itemmean reward ---
# model-free item-level reward: shrunk mean played ratio of the track in TRAIN (same scale as the ratio reward)
if "itemmean" not in REWARDS:
    REWARDS.append("itemmean")

_combine_orig = combine
_token_stats_orig = token_stats


def combine(P, tok, preset, len_sec, len_bias):
    if preset == "itemmean":
        return len_bias[tok]
    return _combine_orig(P, tok, preset, len_sec, len_bias)


def token_stats(ev, V, reward="ratio"):
    if reward == "itemmean":
        return _token_stats_orig(ev, V, "itemdebiased")      # its second output is the shrunk item mean
    return _token_stats_orig(ev, V, reward)
'''
p.write_text(s, encoding="utf-8")
print("patched rl_lib.py")