"""Run each student stage to completion before launching its successor."""
import os
import subprocess
import sys

from experiment_paths import CODE, lock_sources, phase

STAGES = (
    ("train", "phase8_train_student.py", True),
    ("reviews", "phase8_student_reviews.py", True),
    ("review-replay", "phase8_verify_reviews.py", False),
    ("blind", "phase8_rank_blind.py", False),
    ("score", "phase8_score_student.py", False),
    ("accept", "phase8_accept.py", False),
)


def main():
    for stage, script, gpu in STAGES:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0" if gpu else "",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
        command = [sys.executable, str(CODE / "launch_stage.py"), "--wait",
                   str(phase(8)), stage, "p8v1", "--", sys.executable, "-u", str(CODE / script)]
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    lock_sources(8)
    main()
