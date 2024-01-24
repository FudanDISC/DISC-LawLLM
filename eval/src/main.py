import argparse
import os
from functools import partial

import pandas as pd
from ml3m.base import ResponseGenerator
from tabulate import tabulate

from eval import McqRegexEvaluator, QaOpenaiEvaluator
from models import MODELS, get_model
from utils import colored, generate_and_evaluate, get_paths, print_section


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--model_name",
        choices=MODELS.keys(),
        required=True,
        help="the model to evaluate",
    )
    parser.add_argument(
        "--tasks",
        choices=["mcq_sing", "mcq_mult", "qa"],
        nargs="+",
        required=True,
        help="the evaluation tasks",
    )
    parser.add_argument(
        "--n-shot",
        type=int,
        default=4,
        help="the number of few-shot examples (not used in qa)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="the maximum number of iterations for each dataset",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="the verbosity level",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="only evaluate generated results without loading the model"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="only generate the results without evaluating them"
    )
    parser.add_argument(
        "--unique-dir",
        action="store_true",
        help="create distinct directory for each model's result rather than put them together"
    )
    args = parser.parse_args()
    assert not(args.generate_only and args.eval_only)

    # Load the specified model
    model_name = args.model_name
    model = get_model(model_name)

    # Preparation work
    dirname = os.path.dirname(__file__)
    basedir = os.path.join(dirname, "..")
    openai_config = os.path.join(basedir, "..", "openai.json")

    def mcq_info_func(data_item, multi):
        """Data item is a pandas Series."""
        question, A, B, C, D = data_item[["input", "A", "B", "C", "D"]]
        options_repr = "\n".join(
            [f"{label}. {option}" for label, option in zip("ABCD", [A, B, C, D])]
        )
        return model.mcq_formatter(multi=multi, n_shot=args.n_shot).format(
            question=question, options=options_repr
        )

    def qa_info_func(data_item):
        """Data item is a dict."""
        question = data_item["input"]
        return model.qa_formatter().format(question=question)

    ###################################################################################
    #                                                                                 #
    #                Multiple choice questions (single correct option)                #
    #                                                                                 #
    ###################################################################################

    mcq_sing_scores = {}
    if "mcq_sing" in args.tasks:
        mcq_sing_dataset_names = ["cpa", "lbk", "nje", "pae", "pfe", "ungee"]
        for dataset_name in mcq_sing_dataset_names:
            orig_dataset, dataset, save_path = get_paths(
                basedir, "mcq_sing", "csv", dataset_name, model_name, args.unique_dir
            )

            mcq_sing_scores[dataset_name] = generate_and_evaluate(
                task_name="MCQ::sing",
                dataset_name=dataset_name,
                generator_klass=ResponseGenerator,
                generator_kwargs=dict(
                    orig_dataset=orig_dataset,
                    dataset=dataset,
                    info_func=partial(mcq_info_func, multi=False),
                    query_func=model.achat
                    if model_name.startswith("gpt")
                    else model.chat,
                    response_name=f"{model_name}_response",
                    fmt="csv",
                    n_workers=2 if model_name.startswith("gpt") else 1,
                    verbose=args.verbose,
                ),
                evaluator_klasses=[McqRegexEvaluator],
                evaluator_kwargses=[
                    dict(
                        dataset=dataset,
                        save_path=save_path,
                        subjects=["regex_score"],
                        response_name=f"{model_name}_response",
                        verbose=args.verbose,
                    ),
                ],
                max_iter=args.max_iter,
                eval_only=args.eval_only,
                generate_only=args.generate_only,
            )

    ###################################################################################
    #                                                                                 #
    #              Multiple choice questions (multiple correct options)               #
    #                                                                                 #
    ###################################################################################

    mcq_mult_scores = {}
    if "mcq_mult" in args.tasks:
        mcq_mult_dataset_names = ["cpa", "nje", "pae", "ungee"]
        for dataset_name in mcq_mult_dataset_names:
            orig_dataset, dataset, save_path = get_paths(
                basedir, "mcq_mult", "csv", dataset_name, model_name, args.unique_dir
            )
            mcq_mult_scores[dataset_name] = generate_and_evaluate(
                task_name="MCQ::mult",
                dataset_name=dataset_name,
                generator_klass=ResponseGenerator,
                generator_kwargs=dict(
                    orig_dataset=orig_dataset,
                    dataset=dataset,
                    info_func=partial(mcq_info_func, multi=True),
                    query_func=model.achat
                    if model_name.startswith("gpt")
                    else model.chat,
                    response_name=f"{model_name}_response",
                    fmt="csv",
                    n_workers=5 if model_name.startswith("gpt") else 1,
                    verbose=args.verbose,
                ),
                evaluator_klasses=[McqRegexEvaluator],
                evaluator_kwargses=[
                    dict(
                        dataset=dataset,
                        save_path=save_path,
                        subjects=["regex_score"],
                        response_name=f"{model_name}_response",
                        verbose=args.verbose,
                    ),
                ],
                max_iter=args.max_iter,
                eval_only=args.eval_only,
                generate_only=args.generate_only,
            )

    ###################################################################################
    #                                                                                 #
    #                               Question-answering                                #
    #                                                                                 #
    ###################################################################################

    qa_scores = {}
    if "qa" in args.tasks:
        qa_dataset_names = ["short_answer"]

        for dataset_name in qa_dataset_names:
            orig_dataset, dataset, save_path = get_paths(
                basedir, "qa", "json", dataset_name, model_name, args.unique_dir
            )
            qa_scores[dataset_name] = generate_and_evaluate(
                task_name="QA",
                dataset_name=dataset_name,
                generator_klass=ResponseGenerator,
                generator_kwargs=dict(
                    orig_dataset=orig_dataset,
                    dataset=dataset,
                    info_func=qa_info_func,
                    query_func=model.achat
                    if model_name.startswith("gpt")
                    else model.chat,
                    response_name=f"{model_name}_response",
                    fmt="json",
                    n_workers=5 if model_name.startswith("gpt") else 1,
                    verbose=args.verbose,
                ),
                evaluator_klasses=[QaOpenaiEvaluator],
                evaluator_kwargses=[
                    dict(
                        dataset=dataset,
                        save_path=save_path,
                        openai_config=openai_config,
                        info_func=lambda data_item: (
                            data_item["input"],
                            data_item[f"{model_name}_response"],
                            data_item["output"],
                        ),
                        fmt="json",
                        # setting="You are a professional in Chinese law.",
                        verbose=args.verbose,
                    ),
                ],
                max_iter=args.max_iter,
                eval_only=args.eval_only,
                generate_only=args.generate_only,
            )

    ###################################################################################
    #                                                                                 #
    #                            Summarization of results                             #
    #                                                                                 #
    ###################################################################################

    for name, scores in zip(
        ["MCQ::sing", "MCQ::mult", "QA"], [mcq_sing_scores, mcq_mult_scores, qa_scores]
    ):
        if len(scores) == 0:
            continue
        df = pd.DataFrame(scores).T
        df.loc[colored("AVG", "green"), :] = df.mean()
        print_section(f"[{name}] EVALUATION SUMMARY", "green")
        print(tabulate(df, headers="keys", tablefmt="fancy_outline"))
