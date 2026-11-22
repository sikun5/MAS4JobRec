"""Prepare run-local inputs and finish the Phase 5/6 handoffs formerly made by shell commands."""
import argparse
import hashlib
import json
import subprocess

from experiment_paths import ROOT, lock_sources, phase

GUIDE = ROOT / "global_guide/v2_frozen"
MATRIX = GUIDE / "acceptance_result_matrix.md"
MODEL = ROOT / "model/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218"
FOLDERS = {4: ("base",), 5: ("base", "analysis"), 6: ("eval/main",),
           7: ("distill",), 8: ("distill", "student", "eval/student")}


def pin(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Previously recorded input differs: {path}")
    else:
        path.write_bytes(content)


def verify(directory, manifest):
    result = subprocess.run(["sha256sum", "-c", str(manifest.resolve())],
                            cwd=directory, capture_output=True, check=True)
    return result.stdout


def prepare(number):
    run = phase(number)
    evidence = run / "evidence"
    for name in FOLDERS[number]:
        (run / name).mkdir(parents=True, exist_ok=True)
    if number in (4, 8):
        check = verify(MODEL, phase(1) / "env/model_files.sha256")
        pin(run / ("model_check.log" if number == 4 else "evidence/model_verified.txt"), check)
    for upstream in {4: (), 5: (4,), 6: (5,), 7: (6,), 8: (6, 7)}[number]:
        check = verify(phase(upstream), phase(upstream) / f"phase{upstream}.sha256")
        pin(evidence / f"phase{upstream}_before.txt", check)
    if number == 5:
        protocol = (GUIDE / "phase5_trust_and_ranking.md").read_bytes()
        original, marker, _ = protocol.partition("\n## 4. 补充实验结果｜跨 Agent 相关错误分析\n".encode())
        if not marker:
            raise ValueError("Missing frozen Phase 5 protocol boundary")
        pin(evidence / "protocol_before.md", original)
        pin(evidence / "protocol.md", protocol)
    if number == 6:
        log = run / "work_log.md"
        if not log.exists():
            log.write_text("# Phase 6 运行记录\n\n上游 Phase 5 清单已核验。"
                           "各步骤的命令、起止时间和退出状态见 jobs/。\n")
        for name in ("phase6_evaluation.md", "experimental_setting.md", MATRIX.name):
            pin(evidence / name, (GUIDE / name).read_bytes())
    if number == 7:
        pin(evidence / "git_head.txt", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT))
    if number == 8:
        if hashlib.sha256(MATRIX.read_bytes()).hexdigest() != read_json(phase(6) / "acceptance.json")["matrix_sha256"]:
            raise ValueError("Phase 8 requires the accepted Phase 6 matrix at the guide path")
        pin(evidence / "matrix_before.md", MATRIX.read_bytes())


def read_json(path):
    return json.loads(path.read_text())


def verify_upstream(number):
    upstream = number - 1
    evidence = phase(number) / "evidence"
    check = verify(phase(upstream), phase(upstream) / f"phase{upstream}.sha256")
    if check != (evidence / f"phase{upstream}_before.txt").read_bytes():
        raise ValueError("Upstream verification changed since preparation")
    pin(evidence / f"phase{upstream}_after.txt", check)


def close(number):
    run = phase(number)
    accepted = read_json(run / "acceptance.json")
    if accepted["status"] != "passed":
        raise ValueError(f"Missing passing acceptance in {run}")
    verify_upstream(number)
    evidence = run / "evidence"
    if number == 5:
        elapsed = {}
        for stage in ("formal", "supplement", "verify"):
            status = read_json(run / "jobs" / stage / "status.json")
            if status["state"] != "done" or status["exit_code"] != 0:
                raise ValueError(f"Unfinished stage: {stage}")
            elapsed[stage] = status["ended_epoch"] - status["started_epoch"]
        pin(evidence / "run_receipt.json", json.dumps({"job_elapsed_s": elapsed,
            "gpu_compute_hours": 0}, indent=2).encode() + b"\n")
    else:
        pin(run / "phase6_acceptance.md", (
            "# Phase 6 主实验验收\n\n"
            f"通过。共核对 {accepted['statistics']['users']} 名用户、"
            f"{accepted['statistics']['methods']} 种方法。\n\n"
            "[统计复核](evidence/statistics_verified.json) · [评分回执](evidence/score_receipt.json)"
            " · [完整结果](acceptance_result_matrix.md)\n").encode())
    # Keep immutable evidence and completed Phase 5 job receipts. Launcher locks
    # and the live job running this closeout must never enter the seal.
    files = [p for name in (*FOLDERS[number], "evidence") for p in (run / name).rglob("*") if p.is_file()]
    files += [run / "acceptance.json", run / f"phase{number}_acceptance.md", run / "work_log.md"]
    if number == 5:
        files += [run / "jobs" / s / f for s in elapsed for f in ("status.json", "stage.log")]
    else:
        files.append(run / MATRIX.name)
    manifest = run / f"phase{number}.sha256"
    entries = []
    for path in sorted(files):
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
            content_hash = digest.hexdigest()
        entries.append(f"{content_hash}  {path.relative_to(run)}\n")
    # Verify before publishing the filename that marks a sealed run.
    pending = manifest.with_suffix(".pending")
    pending.write_text("".join(entries))
    verify(run, pending)
    pending.replace(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify-upstream", "close"))
    parser.add_argument("phase", type=int, choices=range(4, 9))
    args = parser.parse_args()
    if args.operation != "prepare" and args.phase not in (5, 6):
        parser.error("verify-upstream and close apply to phases 5 and 6")
    lock_sources(args.phase)
    if args.operation == "prepare":
        prepare(args.phase)
    elif args.operation == "verify-upstream":
        verify_upstream(args.phase)
    else:
        close(args.phase)


if __name__ == "__main__":
    main()
