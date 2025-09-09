# System and utilities
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn (Data Prep & Evaluation)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    # Regression
    r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
)

# Linear Models
from sklearn.linear_model import (
    LinearRegression, ElasticNet, LogisticRegression
)

# Tree-based Models
from sklearn.ensemble import (
    # Regression
    RandomForestRegressor, GradientBoostingRegressor,
)

# Advanced Gradient Boosting
from xgboost import XGBRegressor
