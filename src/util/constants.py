
#####################
###     KERAS     ###
#####################
BATCH_SIZE = 10
NUM_CLASSES = 2
EPOCHS = 100
LEARNING_RATE = 0.001

#####################
### RANDOM FOREST ###
#####################
N_ESTIMATORS = 500

TRAIN_FILES = {
    "bikes": "data/bikes.csv",
    "weather": "data/weather.csv",
    "holidays": "data/holidays.csv"
}

TOTAL_TRAINING = "data/train.csv"

TESTING = "data/Testing.csv"

FINAL_TEST = "data/Final Test.csv"