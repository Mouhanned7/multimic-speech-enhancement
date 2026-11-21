"""Run the multi-microphone speech enhancement demo.

Examples:
    python src/run_demo.py                      # synthetic voices, default scene
    python src/run_demo.py --sweep              # + study of the number of mics
    python src/run_demo.py --target a.wav --interferer b.wav
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

from beamformers import run_all
from metrics import si_sdr
from simulate import load_wav, make_scene, synth_voice

FS = 16000


def get_signals(args):
    if args.target:
        target = load_wav(args.target, FS, args.duration)
    else:
        target = synth_voice(args.duration, FS, f0=120, seed=1)
    if args.interferer:
        interf = load_wav(args.interferer, FS, args.duration)
    else:
        interf = synth_voice(args.duration, FS, f0=210, seed=2)
    return target, interf


def evaluate(scene, theta_target):
    outputs = run_all(scene, theta_target)
    ref = scene["reference"]
    scores = {name: si_sdr(sig, ref) for name, sig in outputs.items()}
    return outputs, scores


def save_audio(outputs, out_dir):
    for name, sig in outputs.items():
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fname = fname.replace("+", "plus").replace(".", "") + ".wav"
        sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.9
        wavfile.write(os.path.join(out_dir, fname), FS, (sig * 32767).astype(np.int16))


def plot_spectrograms(outputs, reference, scores, path):
    items = [("Clean target", reference)] + list(outputs.items())
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for ax, (name, sig) in zip(axes.ravel(), items):
        f, t, S = spectrogram(sig, FS, nperseg=512, noverlap=384)
        ax.pcolormesh(t, f, 10 * np.log10(S + 1e-10), vmin=-100, vmax=-30,
                      shading="auto", cmap="magma")
        label = name if name not in scores else "%s (%.1f dB)" % (name, scores[name])
        ax.set_title(label, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Hz")
    for ax in axes[1]:
        ax.set_xlabel("s")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def sweep_mics(target, interf, args, path_csv, path_png):
    rows = []
    for m in range(1, 9):
        scene = make_scene(target, interf, FS, n_mics=m, spacing=args.spacing,
                           theta_target=args.theta_target,
                           theta_interf=args.theta_interf, snr_db=args.snr,
                           sir_db=args.sir, seed=args.seed)
        _, s = evaluate(scene, args.theta_target)
        rows.append([m] + [round(float(s[k]), 2) for k in s if k != "Mic 0 (input)"])
    methods = ["Delay-and-sum", "MPDR", "MVDR (oracle cov.)", "MVDR + post-filter"]
    with open(path_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_mics"] + methods)
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, name in enumerate(methods):
        ax.plot([r[0] for r in rows], [r[i + 1] for r in rows], marker="o", label=name)
    ax.set_xlabel("Number of microphones")
    ax.set_ylabel("SI-SDR (dB)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_png, dpi=130)
    plt.close(fig)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", help="wav file for the target talker")
    p.add_argument("--interferer", help="wav file for the interfering talker")
    p.add_argument("--mics", type=int, default=4)
    p.add_argument("--spacing", type=float, default=0.05, help="mic spacing (m)")
    p.add_argument("--theta-target", type=float, default=0.0, help="degrees")
    p.add_argument("--theta-interf", type=float, default=50.0, help="degrees")
    p.add_argument("--snr", type=float, default=10.0, help="target-to-noise (dB)")
    p.add_argument("--sir", type=float, default=0.0, help="target-to-interferer (dB)")
    p.add_argument("--duration", type=float, default=6.0, help="seconds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--sweep", action="store_true", help="vary the number of mics")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    target, interf = get_signals(args)
    scene = make_scene(target, interf, FS, n_mics=args.mics, spacing=args.spacing,
                       theta_target=args.theta_target,
                       theta_interf=args.theta_interf, snr_db=args.snr,
                       sir_db=args.sir, seed=args.seed)
    outputs, scores = evaluate(scene, args.theta_target)

    base = scores["Mic 0 (input)"]
    lines = ["| Method | SI-SDR (dB) | Gain vs input (dB) |", "|---|---|---|"]
    for name, v in scores.items():
        lines.append("| %s | %.2f | %+.2f |" % (name, v, v - base))
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(args.out, "results.md"), "w") as fh:
        fh.write(table + "\n")

    wavfile.write(os.path.join(args.out, "clean_target.wav"), FS,
                  (scene["reference"] / np.max(np.abs(scene["reference"])) * 0.9
                   * 32767).astype(np.int16))
    save_audio(outputs, args.out)
    plot_spectrograms(outputs, scene["reference"], scores,
                      os.path.join(args.out, "spectrograms.png"))
    if args.sweep:
        rows = sweep_mics(target, interf, args,
                          os.path.join(args.out, "sweep_mics.csv"),
                          os.path.join(args.out, "sweep_mics.png"))
        print("\nSweep over the number of microphones (SI-SDR, dB):")
        for r in rows:
            print(r)
    print("\nFiles written to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
