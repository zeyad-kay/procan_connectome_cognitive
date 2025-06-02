import pandas as pd
import wandb
import ast
import dotenv
from datetime import datetime


def parse_wandb_results(project_results_csv: str) -> pd.DataFrame:
    df = pd.read_csv(project_results_csv)
    df = df.set_index("id")
    config_df = df["config"].apply(lambda x: pd.json_normalize(ast.literal_eval(x)))
    config_df = pd.concat(config_df.to_dict())
    config_df.index = df.index
    summary_df = df["summary"].apply(lambda x: pd.json_normalize(ast.literal_eval(x)))
    summary_df = pd.concat(summary_df.to_dict())
    summary_df.index = df.index
    combined_df = pd.concat(
        [
            df,
            config_df,
            summary_df,
        ],
        axis=1,
    )
    sorted_cols = sorted(combined_df.columns)
    combined_df = combined_df[sorted_cols]
    return combined_df


if __name__ == "__main__":
    dotenv.load_dotenv("../../.env")
    start_time = datetime.now()
    SWEEP = "ez9o8lkz"
    print("Downloading runs...")
    # api = wandb.Api(timeout=30)
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs("zeyadk-university-of-calgary-in-alberta/procan", filters={"sweep": SWEEP})
    # runs = api.runs("zeyadk-university-of-calgary-in-alberta/procan", filters={"$or": [{"sweep": "vly5j7zq"}, {"sweep": "t9vue35k",}, {"sweep": "wvttozxz",}]})
    summary_list, config_list, name_list, state_list, id_list, tags = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    sweep_list = []

    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        id_list.append(run.id)
        state_list.append(run.state)
        sweep_id = None
        if hasattr(run.sweep, "id"):
            sweep_id = run.sweep.id
        sweep_list.append(sweep_id)
        tags.append(run.tags)

    runs_df = pd.DataFrame(
        {
            "summary": summary_list,
            "config": config_list,
            "name": name_list,
            "id": id_list,
            "state": state_list,
            "sweep_id": sweep_list,
            "tags": tags,
        }
    )

    runs_df.to_csv(f"project_raw_data_{SWEEP}.csv")
    print("Raw data download completed and saved to project_raw_data.csv")
    print("Parsing results into human readable format...")

    processed_df = parse_wandb_results(f"project_raw_data_{SWEEP}.csv")
    processed_df.to_csv(f"processed_results_{SWEEP}.csv")
    print(f"Parsing complete. Results saved to processed_results_{SWEEP}.csv")
    end_time = datetime.now() - start_time
    print(f"Script completed in {end_time}")

    # Load with df = pd.read_csv("../processed_results.csv", index_col="id")
