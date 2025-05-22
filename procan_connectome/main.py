import os
import logging
import dotenv
import datetime
import time
import hydra
import pathlib

from procan_connectome.data_ingestion.ingest_data import get_dataset
from procan_connectome.data_processing.pipeline import get_pipeline
from procan_connectome.model_training.model_zoo import get_estimator_and_grid
from procan_connectome.utils.wandb_utils import init_wandb, save_model
from procan_connectome.model_training.loocv_wrapper import LOOCV_Wrapper
from procan_connectome.utils.result_plots import plot_all_figures


def init_logger(cfg):
    logging.basicConfig(
        filename=os.path.join(cfg.paths.logs, log_file_name + "_LOGS"),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    # logging.basicConfig(
    #     filename="./LOGS_TEST",
    #     filemode="a",
    #     format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    #     datefmt="%H:%M:%S",
    #     level=logging.INFO,
    # )
    logger = logging.getLogger(__file__)
    return logger


@hydra.main(
    config_path=f"..{os.sep}configs", config_name="config", version_base="1.4"
)
def main(cfg):
    logger = init_logger(cfg)
    df = get_dataset(cfg.dataset, drop_na=cfg.drop_na, global_only=cfg.global_only, timepoint=cfg.timepoint, cfg.paths)
    exit()
    X, y = df.drop(columns=["Group"]), df["Group"]
    estimator, grid = get_estimator_and_grid(cfg)
    pipe = get_pipeline(cfg, estimator)
    log_file_name = (
        f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}"
        f"_{cfg.model}_{cfg.dataset}"
    )
    run = init_wandb(cfg)

    start = time.time()
    loocv = LOOCV_Wrapper(
        X,
        y,
        estimator,
        pipeline=pipe,
        param_grid=grid,
        label_col="Group",
        log_file_name=log_file_name,
        log_dir=pathlib.Path(cfg.paths.logs),
        verbose=2,
        random_seed=cfg.random_seed,
        **cfg.loocv,
    )
    loocv.fit(X, y)
    end = time.time() - start
    logger.info(f"Process completed in {end//60} minutes.")
    plot_all_figures(
        cfg, loocv.results_df_, loocv.importances_.fillna(0).mean(axis=1).rename("Importance"), loocv.grid_results_
    )
    save_model(cfg, loocv)
    run.finish()


if __name__ == "__main__":
    dotenv.load_dotenv(dotenv_path=".env", override=True)
    main()
