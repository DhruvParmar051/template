from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(num_features, cat_features, 
                       num_strategy="median", cat_strategy="most_frequent"):
    
    # Numeric pipeline
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=num_strategy)),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cat_strategy, fill_value="missing")),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ]
    )
    
    return preprocessor
