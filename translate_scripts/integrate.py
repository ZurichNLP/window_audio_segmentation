import os,sys
import fairseq
from fairseq_cli.generate import get_symbols_to_strip_from_output
import importlib.util
from collections import namedtuple


import ast
import math
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")

# set both vars to False for production environment:
print_info=False
print_debug=False

import time
import torch
import numpy as np

def init():
    global models, generator, tokenizer, task, bpe, cfg, encode_fn, decode_fn, max_positions, use_cuda, src_dict, tgt_dict, align_dict

    parser = fairseq.options.get_interactive_generation_parser()
    args = fairseq.options.parse_args_and_arch(parser)
    cfg = fairseq.dataclass.utils.convert_namespace_to_omegaconf(args)

    logger.info(cfg)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    task = fairseq.tasks.setup_task(cfg.task)

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args = fairseq.checkpoint_utils.load_model_ensemble(
        fairseq.utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = fairseq.utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = fairseq.utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    if cfg.generation.constraints:
        logger.warning(
            "NOTE: Constrained decoding currently assumes a shared subword vocabulary."
        )

    if cfg.interactive.buffer_size > 1:
        logger.info("Sentence buffer size: %s", cfg.interactive.buffer_size)
    logger.info("NOTE: hypothesis and token scores are output in base 2")

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    logger.info("NMT set-up complete")
    if torch.cuda.is_available():
        logger.info("CUDA available.","Current device:",torch.cuda.get_device_name(torch.cuda.current_device())+".", \
            "Number of devices:",torch.cuda.device_count())
    else:
        logger.info("CUDA not available.")
    logger.info("Waiting for texts to translate...")
    return "model"


def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if cfg.generation.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    if cfg.generation.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )


def translate(input_files, prefix=None):
    global models, generator, task, tokenizer, bpe, cfg, encode_fn, decode_fn, max_positions, use_cuda, src_dict, tgt_dict, align_dict

    start_time = time.time()
    total_translate_time = 0

    print("Translate() input: ",input_files)

    start_id = 0
    output_sentences = []
    results = []
    for batch in make_batches(input_files, cfg, task, max_positions, encode_fn):
        bsz = batch.src_tokens.size(0)
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        constraints = batch.constraints
        if use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
            if constraints is not None:
                constraints = constraints.cuda()

        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
        }
        if prefix and cfg.generation.prefix_bias_beta > 0.0:
            print(prefix)
            prefix = bpe.encode(prefix)
            print(prefix)
            prefix = task.target_dictionary.encode_line(prefix, append_eos=False, add_if_not_exist=False,)
            if cfg.generation.prefix_bias_mask >= len(prefix):
                prefix = None
            elif cfg.generation.prefix_bias_mask > 0:
                prefix = prefix[:-cfg.generation.prefix_bias_mask]
                prefix = prefix.cuda()
            print(prefix)

        translate_start_time = time.time()
        translations = task.inference_step(
            generator, models, sample, constraints=constraints, prev_target=prefix
        )
        translate_time = time.time() - translate_start_time
        total_translate_time += translate_time
        list_constraints = [[] for _ in range(bsz)]
        if cfg.generation.constraints:
            list_constraints = [unpack_constraints(c) for c in constraints]
        for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
            src_tokens_i = fairseq.utils.strip_pad(src_tokens[i], tgt_dict.pad())
            constraints = list_constraints[i]
            results.append(
                (
                    start_id + id,
                    src_tokens_i,
                    hypos,
                    {
                        "constraints": constraints,
                        "time": translate_time / len(translations),
                    },
                )
            )

    output_sentences = []
    # sort output to match input order
    for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
        src_str = ''
        if src_dict is not None:
            src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
            print("S-{}\t{}".format(id_, src_str))
            print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
            for constraint in info["constraints"]:
                print(
                    "C-{}\t{}".format(
                        id_, tgt_dict.string(constraint, cfg.common_eval.post_process)
                    )
                )

        # Process top predictions
        for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
            hypo_tokens, hypo_str, alignment = fairseq.utils.post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str=src_str,
                alignment=hypo["alignment"],
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=cfg.common_eval.post_process,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
            )
            detok_hypo_str = decode_fn(hypo_str)
            output_sentences.append(detok_hypo_str)
            score = hypo["score"] / math.log(2)  # convert to base 2
            # original hypothesis (after tokenization and BPE)
            print("H-{}\t{}\t{}".format(id_, score, hypo_str))
            # detokenized hypothesis
            print("D-{}\t{}\t{}".format(id_, score, detok_hypo_str))
            print(
                "P-{}\t{}".format(
                    id_,
                    " ".join(
                        map(
                            lambda x: "{:.4f}".format(x),
                            # convert from base e to base 2
                            hypo["positional_scores"].div_(math.log(2)).tolist(),
                        )
                    ),
                )
            )
            if cfg.generation.print_alignment:
                alignment_str = " ".join(
                    ["{}-{}".format(src, tgt) for src, tgt in alignment]
                )
                print("A-{}\t{}".format(id_, alignment_str))

    return {"result": output_sentences, "time_taken": float('%.6f'%(start_time-time.time()/1000)), "error": None}


import fileinput

if __name__ == "__main__":

    # execute only if run as a script
    m = init()

    for line in fileinput.input():
        line.rstrip()
        print(translate(m,line,None)['result'])
