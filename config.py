import argparse
import json


def load_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to the configuration file"
    )

    # models
    parser.add_argument(
        "--model_gen",
        type=str,
        default="./model_gen",
        help="path to the model to be estimated"
    )
    parser.add_argument(
        "--model_est",
        type=str,
        default="./model_est",
        help="path to the estimator model"
    )
    parser.add_argument(
        "--thread",
        type=int,
        default=1,
        help="number of threads for generation, each thread will use one gpu"
    )

    # data
    parser.add_argument(
        "--data",
        type=str,
        default="./data",
        help="path to the data to be estimated on"
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=1000,
        help="maximum lines to read from data"
    )
    parser.add_argument(
        "--content_length",
        type=int,
        default=1024,
        help="groundtruth text length of generation"
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=256,
        help="prompt length for generation"
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="try to parse data to prompt and content"
    )
    parser.add_argument(
        "--parse_columns",
        type=str,
        default="prompt prompts title content contents",
        help="columns to parse from data"
    )
    parser.add_argument(
        "--parse_seperator",
        type=str,
        default="\n",
        help="seperator appended between columns when parsed as one"
    )

    # generation configs
    parser.add_argument(
        "--gen_max_tokens",
        type=int,
        default=1024,
        help="maximum generation length in tokens"
    )
    parser.add_argument(
        "--gen_min_tokens",
        type=int,
        default=1024,
        help="minimum generation length in tokens"
    )
    parser.add_argument(
        "--gen_top_k",
        type=int,
        default=40,
        help="top k for generation"
    )
    parser.add_argument(
        "--gen_temperature",
        type=float,
        default=0.8,
        help="temperature for generation"
    )
    parser.add_argument(
        "--gen_do_sample",
        action="store_true",
        help="do sample for generation"
    )
    parser.add_argument(
        "--gen_penalty",
        type=float,
        default=1.2,
        help="repetition penalty for generation"
    )
    parser.add_argument(
        "--gen_overwrite",
        action="store_true",
        help="overwrite the existing generation results"
    )
    parser.add_argument(
        "--gen_save_path",
        type=str,
        default=None,
        help="path to save generation results, none for not saving(default)"
    )
    parser.add_argument(
        "--gen_save_stepwise",
        action="store_true",
        help="save generation results stepwise"
    )
    parser.add_argument(
        "--est_token_length",
        type=int,
        default=1024,
        help="maximum token length for logits generation"
    )
    parser.add_argument(
        "--est_overwrite",
        action="store_true",
        help="overwrite the existing logits generation results"
    )
    parser.add_argument(
        "--est_save_path",
        type=str,
        default=None,
        help="path to save logits generation results, none for not saving(default)"
    )
    parser.add_argument(
        "--est_save_stepwise",
        action="store_true",
        help="save logits generation results stepwise"
    )
    parser.add_argument(
        "--log_transform",
        action="store_true",
        help="apply log transform on logits"
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            raw_args = json.loads(f.read()).get("args")
            raw_args = [str(arg) for arg in raw_args]
            args = parser.parse_args(raw_args)
    return args
