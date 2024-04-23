import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
import math
from dataclasses import dataclass, field
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
from sklearn import metrics
import matplotlib.pyplot as plt
from datetime import datetime

# ========== CONFIGURATION CLASS ==========
@dataclass
class ModelConfig:
    predictors: list
    response: str
    train: h2o.H2OFrame
    valid: h2o.H2OFrame
    test: h2o.H2OFrame
    SEED: int
    st: str
    output_file: str
    new_min: int = 5
    new_max: int = 20
    run_time: int = 600
    
# ========== MODEL TRAINER CLASS ========== #
class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.prepare_data()

    def prepare_data(self):
        try:
            # Convert 'Converted' column to factor in all data frames
            for df in [self.config.train, self.config.test, self.config.valid]:
                for col in df.columns:
                    # Check if the column is not of numeric type (integer or float)
                    df[self.config.response] = df[self.config.response].asfactor()
                    if not(df[col].isnumeric()[0]):
                        df[col] = df[col].asfactor()
        except Exception as e:
            raise CustomException(e,sys)

    def train_baseline_model(self):
        try:
            gbm = H2OGradientBoostingEstimator(categorical_encoding="OneHotExplicit")
            gbm.train(x=self.config.predictors, y=self.config.response, training_frame=self.config.train)
            logging.info(f">>>>>>>> BASE MODEL <<<<<<<< \n{gbm}")
            return gbm
        except Exception as e:
            raise CustomException(e,sys)

    def train_hyperparameter_tuned_model(self):
        try:
            # create hyperameter and search criteria lists (ranges are inclusive..exclusive))
            hyper_params_tune = {'max_depth' : list(range(self.config.new_min, self.config.new_max + 1, 1)),
                            'sample_rate': [x/100. for x in range(20,101)],
                            'col_sample_rate' : [x/100. for x in range(20,101)],
                            'col_sample_rate_per_tree': [x/100. for x in range(20,101)],
                            'col_sample_rate_change_per_level': [x/100. for x in range(90,111)],
                            'min_rows': [2**x for x in range(0,int(math.log(self.config.train.nrow,2)-1)+1)],
                            'nbins': [2**x for x in range(4,11)],
                            'nbins_cats': [2**x for x in range(4,13)],
                            'min_split_improvement': [0,1e-8,1e-6,1e-4],
                            'histogram_type': ["UniformAdaptive","QuantilesGlobal","RoundRobin"],
                            'class_sampling_factors': [[(i+1)/10, 1] for i in range(10)],
                            'categorical_encoding': ["OneHotExplicit"],
                            'sample_rate_per_class': [[(i+1)/10, 1] for i in range(10)],
                            'balance_classes': [True]}
            
            search_criteria_tune = {'strategy': "RandomDiscrete",
                            'max_runtime_secs': self.config.run_time,  ## limit the runtime to 60 minutes
                            'max_models': 100,  ## build no more than 100 models
                            'seed' : self.config.SEED,
                            'stopping_rounds' : 5,
                            'stopping_metric' : "AUC",
                            'stopping_tolerance': 1e-3}
            
            gbm_final_grid = H2OGradientBoostingEstimator(distribution='bernoulli',
                        ## more trees is better if the learning rate is small enough 
                        ## here, use "more than enough" trees - we have early stopping
                        ntrees=10000,
                        ## smaller learning rate is better
                        ## since we have learning_rate_annealing, we can afford to start with a 
                        #bigger learning rate
                        learn_rate=0.05,
                        ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                        ## (use 1.00 to disable, but then lower the learning_rate)
                        learn_rate_annealing = 0.99,
                        ## score every 10 trees to make early stopping reproducible 
                        #(it depends on the scoring interval)
                        score_tree_interval = 10)
                
            #Build grid search with previously made GBM and hyper parameters
            final_grid = H2OGridSearch(gbm_final_grid, hyper_params = hyper_params_tune,
                                                grid_id = 'final_grid_' + self.config.st,
                                                search_criteria = search_criteria_tune)
            #Train grid search
            final_grid.train(x = self.config.predictors, 
                    y = self.config.response,
                    ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
                    training_frame = self.config.train,
                    validation_frame = self.config.valid)
            
            ## Sort the grid models by AUC
            sorted_final_grid = final_grid.get_grid(sort_by='auc',decreasing=True)
            logging.info(f"Final Grid: \n{sorted_final_grid}")
            #Get the best model from the list (the model name listed at the top of the table)
            self.best_model = h2o.get_model(sorted_final_grid.sorted_metric_table()['model_ids'][0])
            logging.info(f">>>>>>>>>> BEST MODEL <<<<<<< \n{self.best_model}")
            
            return sorted_final_grid, self.best_model
        except Exception as e:
            raise CustomException(e,sys)
    
    def save_model(self):
        try:
            #h2o.save_model(self.model, path=self.config.output_file, force=True)
            h2o.download_model(self.best_model, path=self.config.output_file)
        except Exception as e:
            raise CustomException(e,sys)

# ========== MODEL EVALUATOR CLASS ==========
import matplotlib.pyplot as plt
import os

class ModelEvaluator:
    def __init__(self, best_model, test_frame, artifact_dir='artifacts'):
        self.best_model = best_model
        self.test_frame = test_frame
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

    def save_plot(self, plot, filename):
        plot_path = os.path.join(self.artifact_dir, filename)
        plt.savefig(plot_path)
        plt.close()

    def evaluate_model(self):
        try:
            # Include logic for model evaluation...
            preds = self.best_model.predict(self.test_frame)
            perf = self.best_model.model_performance(self.test_frame)

            # SHAP Summary Plot
            plt.figure()
            self.best_model.shap_summary_plot(self.test_frame)
            plt.savefig(".artifacts\\shap_summary_plot")
            plt.close()

            # Variable importance plot
            plt.figure()
            self.best_model.varimp_plot()
            plt.savefig("artifacts\\varimp_plot")
            plt.close()

            plt.figure()
            perf.plot(type="pr")
            plt.savefig("artifacts\\pr_plot")
            plt.close()

            plt.figure()
            perf.plot(type="roc")
            plt.savefig("artifacts\\roc_plot")
            plt.close()

            plt.figure()
            self.best_model.gains_lift_plot()
            plt.savefig("artifacts\\gains_lift_plot")
            plt.close()

        except Exception as e:
            raise CustomException(e, sys)
    
    def save_explain_plots(self):
        try:
            obj = self.best_model.explain(self.test_frame, render=False)
            for key in obj.keys():
                logging.info(f"saving {key} plots")
                if not obj.get(key).get("plots"):
                    continue
                plots = obj.get(key).get("plots").keys()

                os.makedirs(f"./artifacts/images/{key}", exist_ok=True)
                for plot in plots:
                    fig = obj.get(key).get("plots").get(plot).figure()
                    fig.savefig(f"./artifacts/images/{key}/{plot}.png")
        except Exception as e:
            raise CustomException(e, sys)

# ========== CUTOFF FINDER CLASS ==========
class CutoffFinder:
    def __init__(self, test_frame, best_model):
        self.test_frame = test_frame
        self.best_model = best_model

    def find_optimal_cutoff(self):
        # Include logic for finding the optimal cutoff...
        pass



