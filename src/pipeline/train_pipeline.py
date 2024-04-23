import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, load_object

from dataclasses import dataclass, field
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
from sklearn import metrics
import matplotlib.pyplot as plt
import datetime
import time

from src.components.train_model import ModelConfig
from src.components.train_model import ModelTrainer
from src.components.train_model import ModelEvaluator
from src.components.train_model import CutoffFinder

# ========== MAIN EXECUTION FLOW ==========
def main():
    try:
        # Initialize H2O and set up the configuration
        h2o.init()
        config = ModelConfig(
                predictors = load_object("artifacts\\predictors"),
                response = "Converted",
                train = h2o.import_file("artifacts\\train.csv"),
                valid = h2o.import_file("artifacts\\valid.csv"),
                test = h2o.import_file("artifacts\\test.csv"),
                SEED = 555,
                st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%b%d_%H%M'),
                run_time = 600,
                output_file='./artifacts/lead_model'
                )

        trainer = ModelTrainer(config)
        logging.info(">>>>>> Started TRAINING for Base Model <<<<<<")
        baseline_model = trainer.train_baseline_model()
        logging.info(">>>>>> Started TRAINING for Hyperparameter Based Model <<<<<<")
        final_grid, tuned_model = trainer.train_hyperparameter_tuned_model()
        trainer.save_model()

        logging.info(">>>>>> Started EVALUATION For Hyperparameter Based Model <<<<<<")
        evaluator = ModelEvaluator(tuned_model, 
                                   config.test,
                                   './artifacts')
        evaluator.evaluate_model()
        #evaluator.save_explain_plots()

        cutoff_finder = CutoffFinder(config.test, tuned_model)
        cutoff_finder.find_optimal_cutoff()
    except Exception as e:
            raise CustomException(e,sys) 

if __name__ == "__main__":
    logging.info(">>>>>> TRAINING PIPELINE BEGINS <<<<<<")
    main()
    logging.info(">>>>>> TRAINING PIPELINE COMPLETED <<<<<<")