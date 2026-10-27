import os
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from carbon_x.sft_utils import load_sft_datasets, resolve_checkpoint


class DatasetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.script = SimpleNamespace(
            dataset_name="example/data", dataset_config="default", dataset_streaming=False,
            dataset_train_split="train[:32]", dataset_test_split="test[:8]",
        )

    @patch("carbon_x.sft_utils.load_dataset")
    def test_train_only_preserves_split_selector_and_example_kwargs(self, load):
        row = {"messages": [], "chat_template_kwargs": {"tokenizer_kwargs": {"auto_dna_tags": False}}}
        load.return_value = [row]
        train, evaluation = load_sft_datasets(self.script, SimpleNamespace(eval_strategy="no", do_eval=False))
        self.assertIs(train, load.return_value)
        self.assertIs(train[0], row)
        self.assertIsNone(evaluation)
        load.assert_called_once_with(path="example/data", name="default", streaming=False, split="train[:32]")

    @patch("carbon_x.sft_utils.load_dataset")
    def test_evaluation_is_loaded_for_strategy_or_final_evaluation(self, load):
        for strategy, do_eval in [("steps", False), ("no", True)]:
            with self.subTest(strategy=strategy, do_eval=do_eval):
                load.reset_mock()
                load.side_effect = ["training", "evaluation"]
                self.assertEqual(
                    load_sft_datasets(self.script, SimpleNamespace(eval_strategy=strategy, do_eval=do_eval)),
                    ("training", "evaluation"),
                )
                self.assertEqual(load.call_args.kwargs["split"], "test[:8]")


class CheckpointTests(unittest.TestCase):
    def test_explicit_checkpoint_wins_over_newer_discovered_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            for step in (2, 3):
                Path(directory, f"checkpoint-{step}").mkdir()
            args = SimpleNamespace(output_dir=directory, resume_from_checkpoint="checkpoint-2")
            self.assertEqual(resolve_checkpoint(args), "checkpoint-2")
            args.resume_from_checkpoint = None
            self.assertEqual(resolve_checkpoint(args), str(Path(directory, "checkpoint-3")))
            args.overwrite_output_dir = True
            self.assertIsNone(resolve_checkpoint(args))

    def test_new_output_directory_starts_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=str(Path(directory, "new")), resume_from_checkpoint=None)
            self.assertIsNone(resolve_checkpoint(args))


@unittest.skipUnless(os.environ.get("CARBON_SFT_INTEGRATION") == "1", "Requires authenticated Hub access")
class TokenizerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from trl import ModelConfig
        from carbon_x.model_utils import get_tokenizer

        cls.tokenizer = get_tokenizer(
            ModelConfig(model_name_or_path="HuggingFaceBio/Carbon-X-Base"),
            SimpleNamespace(trust_remote_code=True, chat_template=None),
        )
        cls.tokenizer.eos_token = "<|im_end|>"

    def test_assistant_mask_and_per_example_dna_tagging(self):
        tokenizer = self.tokenizer
        messages = [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Describe ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC."},
            {"role": "assistant", "content": "This is a DNA sequence."},
            {"role": "user", "content": "How long is it?"},
            {"role": "assistant", "content": "Count its bases."},
        ]
        encoded = []
        for auto_tags in (False, True):
            result = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                tokenizer_kwargs={"auto_dna_tags": auto_tags},
            )
            ids, mask = result["input_ids"], result["assistant_masks"]
            self.assertEqual(len(ids), len(mask))
            self.assertIn(1, mask)
            self.assertIn(0, mask)
            assistant = tokenizer.decode([token for token, selected in zip(ids, mask) if selected])
            self.assertIn("This is a DNA sequence.", assistant)
            self.assertIn("Count its bases.", assistant)
            self.assertNotIn("Answer briefly", assistant)
            self.assertNotIn("Describe", assistant)
            self.assertNotIn("How long", assistant)
            self.assertIn(tokenizer.eos_token_id, [token for token, selected in zip(ids, mask) if selected])
            encoded.append(ids)
        self.assertNotIn(tokenizer.dna_begin_token_id, encoded[0])
        self.assertIn(tokenizer.dna_begin_token_id, encoded[1])

    def test_trl_preprocessing_and_padding_on_real_data(self):
        from datasets import load_dataset
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from trl import SFTTrainer
        from carbon_x.configs import SFTConfig

        tokenizer = self.tokenizer
        dataset = load_dataset("HuggingFaceBio/FineDNA-SFT", split="train[:32]")
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
        ))
        with tempfile.TemporaryDirectory() as directory:
            trainer = SFTTrainer(
                model=model, processing_class=tokenizer, train_dataset=dataset,
                args=SFTConfig(
                    output_dir=directory, use_cpu=True, bf16=False, report_to="none",
                    max_length=9984, assistant_only_loss=True, eos_token="<|im_end|>",
                    per_device_train_batch_size=2, dataset_num_proc=1,
                ),
            )
            for row in trainer.train_dataset:
                self.assertLessEqual(len(row["input_ids"]), 9984)
                self.assertTrue(any(label != -100 for label in row["labels"]))
            for i in (0, 1):
                original = tokenizer.apply_chat_template(
                    dataset[i]["messages"], tokenize=True, return_dict=True,
                    return_assistant_tokens_mask=True, **dataset[i]["chat_template_kwargs"],
                )
                expected = [token if selected else -100 for token, selected in
                            zip(original["input_ids"], original["assistant_masks"])][:9984]
                self.assertEqual(trainer.train_dataset[i]["labels"], expected)
            batch = trainer.data_collator([trainer.train_dataset[i] for i in (0, 1)])
            self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())

    def test_entrypoint_trains_evaluates_saves_and_resumes_on_cpu(self):
        from datasets import Dataset
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from trl import ModelConfig, ScriptArguments
        from carbon_x.configs import SFTConfig

        entrypoint = Path(__file__).resolve().parents[1] / "training/posttraining/sft.py"
        main = runpy.run_path(str(entrypoint))["main"]
        dataset = Dataset.from_list([
            {"messages": [{"role": "user", "content": "Name a DNA base."},
                          {"role": "assistant", "content": base}]}
            for base in ("Adenine", "Thymine")
        ])

        def tiny_model(*_):
            return Qwen3ForCausalLM(Qwen3Config(
                vocab_size=len(self.tokenizer), hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            ))

        with tempfile.TemporaryDirectory() as directory, patch.dict(main.__globals__, {
            "get_model": tiny_model, "get_tokenizer": lambda *_: self.tokenizer,
            "load_sft_datasets": lambda *_: (dataset, dataset),
        }):
            for max_steps in (1, 2):
                args = SFTConfig(
                    output_dir=directory, use_cpu=True, bf16=False, report_to="none", optim="adamw_torch",
                    max_length=128, assistant_only_loss=True, eos_token="<|im_end|>",
                    max_steps=max_steps, per_device_train_batch_size=1, logging_steps=1,
                    do_eval=True, save_strategy="steps", save_steps=1, dataset_num_proc=1,
                )
                main(ScriptArguments(dataset_name="tiny-fixture"), args, ModelConfig())
                state = json.loads(Path(directory, "trainer_state.json").read_text())
                self.assertEqual(state["global_step"], max_steps)
                import torch

                optimizer = torch.load(
                    Path(directory, f"checkpoint-{max_steps}", "optimizer.pt"), weights_only=True, map_location="cpu",
                )
                self.assertTrue(optimizer["state"])
                self.assertTrue(all(float(value["step"]) == max_steps for value in optimizer["state"].values()))
            self.assertTrue(Path(directory, "eval_results.json").is_file())
            config = json.loads(Path(directory, "config.json").read_text())
            self.assertTrue(config["use_cache"])
            generation = json.loads(Path(directory, "generation_config.json").read_text())
            self.assertEqual(generation["eos_token_id"], self.tokenizer.eos_token_id)
            from transformers import AutoModelForCausalLM, AutoTokenizer

            reloaded_tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True)
            reloaded_model = AutoModelForCausalLM.from_pretrained(directory)
            self.assertEqual(reloaded_tokenizer.eos_token_id, reloaded_model.generation_config.eos_token_id)


if __name__ == "__main__":
    unittest.main()
