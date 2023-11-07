# French Trot Horse Racing Forecasting Competition Submission

Team Members:
1) Hassan Khurram - University of Toronto, Engineering Science (Machine Intelligence Major)
2) Nicholas Susanto - University of Toronto, Statistics Specialist
3) Justin Alianto - University of Toronto, Computer Science Specialist

## Instructions to Run Model
Our model script lies inside horse_eda_model.ipynb. In order to run the model with a new dataset, just change the data filepath inside the third cell of the entire notebook with the one that you want to use and run the whole script from start to finish in one go.

Note the following:
- Be mindful to add your own new dataset into the same folder as the model script.
- Remember to have xgboost installed inside your Python environment (via Anaconda, etc) in your Jupyter notebook. The installation code is commented out in the second cell.

## EDA
We first constructed a correlation matrix, and we were able to remove some of the redundant columns from our selected set of features that contained information about the target label that we chose (see the section below). This includes the performance columns which are highly correlated with each other.

## Data Prep & Cleaning
### Missing Values
Based on the exploration steps above, we saw that column with blanks (with the number of blank observations):
1) AgeRestriction = 633
2) ClassRestriction = 29588
3) CourseIndicator = 1068301
4) HandicapType = 1042792
5) SexRestriction = 888663
6) RaceGroup = 1177555 (Level of competition, G1 being the best; might be useful to prediction)

Based on this, we removed RaceGroup, CourseIndicator, HandicapType, SexRestriction, AgeRestriction and ClassRestriction from our model.

### Adding Performance Predictors
We defined a new performance metric, cumulative average quantiles of finish position, which indicates the average quantiles a horse finished on all races preceding each race (excluding current race). The new parameter would provide us a cumulative performance of the horse to, hopefully, better predict their finishing position in future races. It takes into account all horses which other performance predictors might not catch.

### Feature Selection & Feature Engineering
Apart from removing the columns with numerous blank values (RaceGroup, CourseIndicator, HandicapType, SexRestriction, AgeRestriction and ClassRestriction), we also removed ID columns, which would have provided too much variance to our data and they would not have suggested any meaningful classifications due to the number of IDs that exist.

Before passing the data into the model for training, we also made sure to convert all categorical features to be of ‘category’ type in Python. The final list of features that we are using are:

Data Columns (total 16 columns):
| #  | Feature Name                | Data Type |
|----|-----------------------------|-----------|
| 0  | Barrier                     | category  |
| 1  | Distance                    | float64   |
| 2  | FrontShoes                  | category  |
| 3  | Gender                      | category  |
| 4  | GoingAbbrev                 | category  |
| 5  | HindShoes                   | category  |
| 6  | HorseAge                    | int64     |
| 7  | RacePrizemoney              | float64   |
| 8  | RacingSubType               | category  |
| 9  | Saddlecloth                 | category  |
| 10 | StartType                   | category  |
| 11 | StartingLine                | int64     |
| 12 | Surface                     | category  |
| 13 | WeightCarried               | float64   |
| 14 | WetnessScale                | int64     |
| 15 | PastAveragePositionQuantile | float64   |

### Setting Up Target Label
We decided to approach the problem through a different perspective. We find that the traditional binary outcome (won or lost) treats any non-first-placing horses the same, regardless of their relative position to the winning horses. Horses who placed in the top-three, for example, should have a better chance of winning future races than those in the last place.

Hence, we predicted the win probability based on each horse’s relative position (position quantiles) in each race. Horses who were disqualified or received violations (BS, UN, etc.) were given the largest quantile value (i.e., last place or 1) to account for the violations received.

## Model Training
To train our ML model to make predictions on new races, we used XGBoost with the following hyperparameters:
params = {
    'learning_rate': 0.7, 
    'eval_metric': 'logloss',
    'max_depth': 5,
    'subsample': 1,
    'colsample_bytree': 0.8,
    'lambda': 0.8,
    'alpha': 0.3
}

## Model Evaluation
To evaluate our model performance, since we are using continuous target labels, we calculated both the mean squared error and the logloss value as mentioned in the competition website.
