import argparse
import json


def load_config(args): 
    args_dict = vars(args)
    parser = get_args_parser()
    with open(args.config, "r") as f:
        raw_args = json.loads(f.read()).get("args")
        raw_args = [str(arg) for arg in raw_args]
        args_new = parser.parse_args(raw_args)
    args_dict.update(vars(args_new))
    args = argparse.Namespace(**args_dict)
    return args

def get_args_parser():
    parser = argparse.ArgumentParser()
    # config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to the configuration file"
    )
    return parser
