import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "training/posttraining/sft.slurm"
SMOKE_LAUNCHER = LAUNCHER.with_name("sft_smoke.slurm")


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / ".venv/bin").mkdir(parents=True)
        (self.root / ".venv/bin/python").symlink_to(sys.executable)
        (self.root / ".venv/bin/accelerate").symlink_to(sys.executable)
        configs = self.root / "training/posttraining/accelerate_configs"
        configs.mkdir(parents=True)
        (configs / "deepspeed_zero3.yaml").write_text("{}")
        (self.root / "recipe.yaml").write_text("{}")
        binaries = self.root / "bin"
        binaries.mkdir()
        for name, body in {
            "scontrol": "#!/bin/bash\nprintf '%s\\n' node0 node1\n",
            "srun": f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n",
        }.items():
            path = binaries / name
            path.write_text(body)
            path.chmod(0o755)
        self.env = {
            **os.environ, "PATH": f"{binaries}:/usr/bin:/bin", "SLURM_NNODES": "2",
            "SLURM_GPUS_ON_NODE": "8", "SLURM_JOB_NODELIST": "node[0-1]", "SLURM_PROCID": "0",
        }
        self.env.pop("BASH_ENV", None)
        for key in list(self.env):
            if key.startswith("BASH_FUNC_"):
                del self.env[key]

    def run_launcher(self, *args):
        return subprocess.run(
            ["bash", str(LAUNCHER), "--config", "recipe.yaml", "--accelerator", "deepspeed_zero3", *args],
            cwd=self.root, env=self.env, text=True, capture_output=True,
        )

    def test_rank_is_evaluated_on_each_node_and_arguments_are_preserved(self):
        result = self.run_launcher("--", "--output_dir", "scratch/space in path", "--gradient_accumulation_steps", "3")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        worker = args[args.index("-c") + 1]
        self.assertIn('--machine_rank "$SLURM_PROCID"', worker)
        self.assertIn("--rdzv_backend static", worker)
        self.assertIn('mkdir -p "$TMPDIR"', worker)
        self.assertEqual(args[-4:], ["--output_dir", "scratch/space in path", "--gradient_accumulation_steps", "3"])
        self.assertIn("16", args)

    def test_legacy_args_support_quotes_and_reject_invalid_quotes(self):
        result = self.run_launcher("--args", '--output_dir "scratch/space in path" --max_steps 3')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)[-4:], ["--output_dir", "scratch/space in path", "--max_steps", "3"])
        self.assertNotEqual(self.run_launcher("--args", '--output_dir "unterminated').returncode, 0)


class SmokeLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / ".venv/bin").mkdir(parents=True)
        # The validator is separate from the orchestration being tested here.
        (self.root / ".venv/bin/python").symlink_to("/usr/bin/true")
        launcher = self.root / "training/posttraining/sft.slurm"
        launcher.parent.mkdir(parents=True)
        launcher.write_text(
            '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
            'printf "#!/bin/bash\\nexit 99\\n" > training/posttraining/sft.slurm\n'
        )
        self.output = self.root / "output"
        self.log = self.root / "calls.log"
        self.env = {
            **os.environ, "SLURM_JOB_ID": "1234", "CARBON_STAGE_VENV": "0",
            "CARBON_SMOKE_OUTPUT_DIR": str(self.output), "CALL_LOG": str(self.log),
            "CARBON_SMOKE_RESUME_ONLY": "0",
        }

    def run_smoke(self):
        return subprocess.run(
            ["bash", str(SMOKE_LAUNCHER)], cwd=self.root, env=self.env,
            capture_output=True, text=True,
        )

    def test_snapshot_survives_repository_launcher_edit_between_phases(self):
        result = self.run_smoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertIn("--max_steps 3", calls[0])
        self.assertIn("--max_steps 4", calls[1])
        self.assertIn(f"--resume_from_checkpoint {self.output}/checkpoint-2", calls[1])

    def test_resume_only_requires_checkpoint_and_skips_first_phase(self):
        self.env["CARBON_SMOKE_RESUME_ONLY"] = "1"
        self.assertNotEqual(self.run_smoke().returncode, 0)
        self.assertFalse(self.log.exists())
        checkpoint = self.output / "checkpoint-2"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}")
        result = self.run_smoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text().splitlines()
        self.assertEqual(len(calls), 1)
        self.assertIn("--max_steps 4", calls[0])


if __name__ == "__main__":
    unittest.main()
