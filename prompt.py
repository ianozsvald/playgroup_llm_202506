import argparse

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

import utils

# shows all the contents in one line
# Please write a 1 sentence description about {{ patterns }}.

# {% for pattern in patterns %}
# Here is an example input and output pattern
# {{ pattern }}
# {% endfor %}


def make_prompt(prompt_name, patterns, target="train", func_dict={}):
    """
    Create a prompt from a file-based template plus patterns.
    """
    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template(prompt_name)
    # add any functions to the template
    template.globals.update(func_dict)
    # only provide the train patters
    prompt = template.render(patterns_input_output=patterns[target])
    return prompt


def get_func_dict():
    func_dict = {  # "write_grid": utils.write_grid,
        "make_grid_plain": utils.make_grid_plain,
        "make_grid_csv_quoted": utils.make_grid_csv_quoted,
        "make_grid_csv_english_words": utils.make_grid_csv_english_words,
        "make_grid_csv": utils.make_grid_csv,
    }
    return func_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)  # read __doc__ attribute
    parser.add_argument(
        "-p",
        "--problem_name",
        type=str,
        nargs="?",
        help="name of an ARC AGI problem e.g. 9565186b"
        " (default: %(default)s))",  # help msg 2 over lines with default
        default="9565186b",
    )
    parser.add_argument(
        "-t",
        "--template_name",
        type=str,
        nargs="?",
        help="template to use in ./templates/",
        default="prompt1.txt",
    )

    args = parser.parse_args()
    print(args)

    func_dict = get_func_dict()
    patterns = utils.get_examples(args.problem_name)
    prompt = make_prompt(
        args.template_name, patterns, target="train", func_dict=get_func_dict()
    )
    print(prompt)
