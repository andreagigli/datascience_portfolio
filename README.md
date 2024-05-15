# Data Science Project: Credit Risk Analysis



## Project Description

This repository provides a credit risk analysis on UCI's [German Credit Data dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
A cleaned version of the dataset is used, presented on [Kaggle](https://www.kaggle.com/datasets/uciml/german-credit/data), which however does not contain the target value.
For this reason, the version of the cleaned Kaggle dataset used is the one provided at [this link](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk).


### Info About the Dataset

The info about the clean dataset is adapted from the original [Kaggle page](https://www.kaggle.com/datasets/uciml/german-credit/data).

> Context.\
> The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by Prof. Hofmann. 
> In this dataset, each entry represents a person who takes a credit by a bank. 
> Each person is classified as good or bad credit risks according to the set of attributes. 
> The link to the original dataset can be found below.
>
> Content.\
> It is almost impossible to understand the original dataset due to its complicated system of categories and symbols. 
> Thus, I wrote a small Python script to convert it into a readable CSV file. 
> Several columns are simply ignored, because in my opinion either they are not important or their descriptions are obscure. 
> The selected attributes are:
> 
> 1. Age (numeric)
> 2. Sex (text: male, female)
> 3. Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
> 4. Housing (text: own, rent, or free)
> 5. Saving accounts (Optional text: little, moderate, quite rich, rich)
> 6. Checking account (Optional text: little, moderate, rich)
> 7. Credit amount (numeric, in German Marks)
> 8. Duration (numeric, in month)
> 9. Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)
> 10. Risk (text: good, bad)


## Try It Out-of-the-Box

You can run an example analysis, where house prices predictions are performed on the California Housing Dataset:

```
cd path/to/portfolio_ML_datascience 
git checkout -b credit_risk_analysis
python analysis_template.py --data_path ../../data/external/gcrdb/ --data_loading_fn load_gcrdb --preprocessing_fn preprocess_gcrdb --eda_fn eda_gcrdb --feature_extraction_fn features_gcrdb --split_fn split_train_test --split_ratio "70 30" --model sklearn_compatible_LGBMClassifier  OR   sklearn_SVC --hparams "{\"sklearn_compatible_LGBMClassifier__n_estimators\": \"randint(100, 500)\"}"  OR  "{\"sklearn_SVC__C\": \"uniform(0.00001, 10)\"}" --hopt_n_rndcv_samplings 3 --hopt_subsampling_fn hopt_subsampling_passthrough --hopt_subsampling_rate 1.0 --n_folds 3 --prediction_fn predict_sklearn --evaluation_fn evaluate_gcrdb --log_level INFO --random_seed 0 --save_output --output_data_dir ../../data/processed/ --output_model_dir ../../models/ --output_reports_dir ../../outputs/reports/ --output_figures_dir ../../outputs/figures/
```
