"""YAML options loader and helper utilities."""

import os
from collections import OrderedDict
import yaml


def ordered_yaml():
    """Support OrderedDict for YAML loading and dumping."""
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse_yaml(yaml_path):
    """Load and parse YAML file into a dict."""
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Configuration YAML not found: {yaml_path}")
    with open(yaml_path, mode="r", encoding="utf-8") as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    return opt


def dict2str(opt, indent_level=1):
    """Format dictionary into a readable multi-line string."""
    msg = "\n"
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_level * 2) + str(k) + ": [\n"
            msg += dict2str(v, indent_level + 1)
            msg += " " * (indent_level * 2) + "]\n"
        else:
            msg += " " * (indent_level * 2) + str(k) + ": " + str(v) + "\n"
    return msg
