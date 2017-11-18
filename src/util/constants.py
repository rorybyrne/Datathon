
#####################
###     KERAS     ###
#####################
BATCH_SIZE = 500
EPOCHS = 10000
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

PCRP_SWAP = {
    "5":"11",
    "6":"12",
    "7":"1"
}