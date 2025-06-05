"""Run a set of problems, report on the results"""

import logging

from dotenv import load_dotenv

import method1_text_prompt
import utils
from config import BREAK_IF_NOT_CHECKED_IN, providers
from litellm_helper import call_llm, check_litellm_key
from prompt import get_func_dict, make_prompt
from run_code import execute_transform
from utils import add_argument_parser, extract_from_code_block, write_grid

logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)


load_dotenv()


if __name__ == "__main__":
    if BREAK_IF_NOT_CHECKED_IN:
        utils.break_if_not_git_committed()

    parser = utils.add_argument_parser(template_name=True, model_name=True)
    args = parser.parse_args()
    print(args)

    logger.info(f"{args=}")

    check_litellm_key(args)

    # Slightly harder problems
    # https://arcprize.org/play?task=1caeab9d
    # https://arcprize.org/play?task=1cf80156
    # https://arcprize.org/play?task=1e0a9b12
    # https://arcprize.org/play?task=1e32b0e9
    # https://arcprize.org/play?task=0607ce86
    # https://arcprize.org/play?task=06df4c85
    # all_problems_to_run = ['1caeab9d', '1cf80156', '1e0a9b12', '1e32b0e9', '0607ce86', '06df4c85']

    # Easier problems
    # https://arcprize.org/play?task=9565186b
    # https://arcprize.org/play?task=0d3d703e
    # https://arcprize.org/play?task=08ed6ac7
    # https://arcprize.org/play?task=0a938d79
    # https://arcprize.org/play?task=178fcbfb
    # https://arcprize.org/play?task=1a07d186
    all_problems_to_run = [
        "9565186b",
        "0d3d703e",
        "08ed6ac7",
        "0a938d79",
        "178fcbfb",
        "1a07d186",
    ]
    result_rr_trains = []  # list of lists of rr_trains for each problem

    for problem_to_run in all_problems_to_run:
        print(f"Running problem: {problem_to_run}")
        # load a single problem
        problems = utils.get_examples(problem_to_run)

        model = args.model_name

        #entry_point = method1_text_prompt.run_experiment_for_iterations
        import method3_reflexion
        entry_point = method3_reflexion.run_experiment
        llm_responses, rr_trains = entry_point(
            model=model,
            provider=providers[args.model_name],
            iterations=1,
            problems=problems,
            template_name=args.template_name,
        )
        result_rr_trains.append(rr_trains)

    for problem_to_run, rr_trains in zip(all_problems_to_run, result_rr_trains):
        assert len(rr_trains) == 1, "Expected exactly one rr_train per problem"
        rr_train = rr_trains[0]
        ran_all_train_problems_correctly = rr_train[
            0
        ].transform_ran_and_matched_for_all_inputs
        ran_at_least_one_train_problem_correctly = rr_train[
            0
        ].transform_ran_and_matched_at_least_once
        indicator = "✅" if ran_all_train_problems_correctly else "❌"
        print(
            f"{indicator} On {problem_to_run} {ran_all_train_problems_correctly=} {ran_at_least_one_train_problem_correctly=}"
        )
