import os
import time


def colored(obj, color):
    if color == "black":
        return f"\033[30m{obj}\033[39m"
    elif color == "red":
        return f"\033[31m{obj}\033[39m"
    elif color == "green":
        return f"\033[32m{obj}\033[39m"
    elif color == "yellow":
        return f"\033[33m{obj}\033[39m"
    elif color == "blue":
        return f"\033[34m{obj}\033[39m"
    elif color == "magenta":
        return f"\033[35m{obj}\033[39m"
    elif color == "cyan":
        return f"\033[36m{obj}\033[39m"
    elif color == "white":
        return f"\033[37m{obj}\033[39m"


def print_section(section_name, color):
    formatted_section = (
        "\n======================================================================\n"
        f"            {section_name}"
        "\n======================================================================\n"
    )
    print(colored(formatted_section, color))


def get_paths(basedir, prefix, suffix, dataset_name, model_name, unique_dir=False):
    if unique_dir:
        return (
            os.path.join(basedir, "datasets", f"{prefix}_{dataset_name}.{suffix}"),
            os.path.join(basedir, "responses", model_name, f"{prefix}_{dataset_name}.{suffix}"),
            os.path.join(basedir, "results", model_name, f"{prefix}_{dataset_name}.csv"),
        )
    return (
        os.path.join(basedir, "datasets", f"{prefix}_{dataset_name}.{suffix}"),
        os.path.join(basedir, "responses", f"{prefix}_{dataset_name}.{suffix}"),
        os.path.join(basedir, "results", f"{prefix}_{model_name}_{dataset_name}.csv"),
    )


def generate_until_completed(generator, max_iter):
    for it in range(max_iter):
        completed = generator.generate()
        if completed:
            break
        if it != max_iter - 1:
            print(colored("Sleeping for 30s", "magenta"), end=" ", flush=True)
            for _ in range(30):
                time.sleep(1)
                print(colored(".", "magenta"), end="", flush=True)
            print()
        
    if not completed:
        raise ValueError


def evaluate_until_completed(evaluator, max_iter):
    for it in range(max_iter):
        completed = evaluator.evaluate()
        if completed:
            break
        if it != max_iter - 1:
            print(colored("Sleeping for 30s", "magenta"), end=" ", flush=True)
            for _ in range(30):
                time.sleep(1)
                print(colored(".", "magenta"), end="", flush=True)
            print()
    if not completed:
        raise ValueError
    return evaluator.load_avg_score()


def generate_and_evaluate(
    task_name,
    dataset_name,
    generator_klass,
    generator_kwargs,
    evaluator_klasses,
    evaluator_kwargses,
    max_iter,
    eval_only=False,
    generate_only=False,
):
    sec_formatter = f"[{task_name}] {{}} RESPONSES FOR {dataset_name.upper()}"

    # Generate responses
    if not eval_only:
        print_section(sec_formatter.format("GENERATING"), "cyan")
        generator = generator_klass(**generator_kwargs)
        generate_until_completed(generator, max_iter=max_iter)

    # Evaluate responses
    avg_scores = {}
    if not generate_only:
        assert len(evaluator_klasses) == len(evaluator_kwargses)
        for evaluator_klass, evaluator_kwargs in zip(evaluator_klasses, evaluator_kwargses):
            print_section(sec_formatter.format("EVALUATING"), "cyan")
            evaluator = evaluator_klass(**evaluator_kwargs)
            avg_scores.update(evaluate_until_completed(evaluator, max_iter=max_iter))
    return avg_scores
