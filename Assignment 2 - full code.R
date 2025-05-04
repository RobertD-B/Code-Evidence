#Loading training data set
adult <- read.csv("adult.data", header=FALSE)

#Labelling the training dataset variables
names(adult) <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income")

#Loading testing data set
adult_test <- read.csv("adult_test.test", header=FALSE)

#Labelling the testing dataset variables
names(adult_test) <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income")

#Loading required packages
library(tidyverse)
library(rpart)
library(caret)
library(randomForest)
library(gbm)
library(dismo)

## Conversion of words into numerical integers that can be used for modelling for training dataset
adult$num_workclass <- case_when(adult$workclass == " Private" ~ 0,
                                 adult$workclass ==  " Self-emp-not-inc" ~ 1,
                                 adult$workclass == " Self-emp-inc" ~ 2,
                                 adult$workclass == " Federal-gov" ~ 3,
                                 adult$workclass == " Local-gov" ~ 4,
                                 adult$workclass == " State-gov" ~ 5,
                                 adult$workclass == " Without-pay" ~ 6,
                                 adult$workclass == " Never-worked" ~ 7)
adult$num_workclass <- as.integer(adult$num_workclass)

adult$num_marital_status <- case_when(adult$marital_status == " Married-civ-spouse" ~ 0,
                                      adult$marital_status == " Divorced" ~ 1,
                                      adult$marital_status == " Never-married" ~ 2,
                                      adult$marital_status == " Separated" ~ 3,
                                      adult$marital_status == " Widowed" ~ 4,
                                      adult$marital_status == " Married-spouse-absent" ~ 5,
                                      adult$marital_status == " Married-AF-spouse" ~ 6,)
adult$num_marital_status <- as.integer(adult$num_marital_status)

adult$num_occupation <- case_when(adult$occupation == " Tech-support" ~ 0,
                                  adult$occupation == " Craft-repair" ~ 1,
                                  adult$occupation == " Other-service" ~ 2,
                                  adult$occupation == " Sales" ~ 3,
                                  adult$occupation == " Exec-managerial" ~ 4,
                                  adult$occupation == " Prof-specialty" ~ 5,
                                  adult$occupation == " Handlers-cleaners" ~ 6,
                                  adult$occupation == " Machine-op-inspct" ~ 7,
                                  adult$occupation == " Adm-clerical" ~ 8,
                                  adult$occupation == " Farming-fishing" ~ 9,
                                  adult$occupation == " Transport-moving" ~ 10,
                                  adult$occupation == " Priv-house-serv" ~ 11,
                                  adult$occupation == " Protective-serv" ~ 12,
                                  adult$occupation == " Armed-Forces" ~ 13)
adult$num_occupation <- as.integer(adult$num_occupation)

adult$num_relationship <- case_when(adult$relationship == " Wife" ~ 0,
                                    adult$relationship == " Own-child" ~ 1,
                                    adult$relationship == " Husband" ~ 2,
                                    adult$relationship == " Not-in-family" ~ 3,
                                    adult$relationship == " Other-relative" ~ 4,
                                    adult$relationship == " Unmarried" ~ 5)
adult$num_relationship <- as.integer(adult$num_relationship)

adult$num_race <- case_when(adult$race == " White" ~ 0,
                            adult$race == " Asian-Pac-Islander" ~ 1,
                            adult$race == " Amer-Indian-Eskimo" ~ 2,
                            adult$race == " Other" ~ 3,
                            adult$race == " Black" ~ 4)
adult$num_race <- as.integer(adult$num_race)

adult$num_sex <- case_when(adult$sex == " Female" ~ 0,
                           adult$sex == " Male" ~ 1)
adult$num_sex <- as.integer(adult$num_sex)

adult$binary_income <- case_when(adult$income == " >50K" ~ 0,
                                 adult$income == " <=50K" ~ 1)

## Conversion of words into numerical integers that can be used for modelling for testing dataset

adult_test$num_workclass <- case_when(adult_test$workclass == " Private" ~ 0,
                                      adult_test$workclass ==  " Self-emp-not-inc" ~ 1,
                                      adult_test$workclass == " Self-emp-inc" ~ 2,
                                      adult_test$workclass == " Federal-gov" ~ 3,
                                      adult_test$workclass == " Local-gov" ~ 4,
                                      adult_test$workclass == " State-gov" ~ 5,
                                      adult_test$workclass == " Without-pay" ~ 6,
                                      adult_test$workclass == " Never-worked" ~ 7)
adult_test$num_workclass <- as.integer(adult_test$num_workclass)

adult_test$num_marital_status <- case_when(adult_test$marital_status == " Married-civ-spouse" ~ 0,
                                           adult_test$marital_status == " Divorced" ~ 1,
                                           adult_test$marital_status == " Never-married" ~ 2,
                                           adult_test$marital_status == " Separated" ~ 3,
                                           adult_test$marital_status == " Widowed" ~ 4,
                                           adult_test$marital_status == " Married-spouse-absent" ~ 5,
                                           adult_test$marital_status == " Married-AF-spouse" ~ 6,)
adult_test$num_marital_status <- as.integer(adult_test$num_marital_status)

adult_test$num_occupation <- case_when(adult_test$occupation == " Tech-support" ~ 0,
                                       adult_test$occupation == " Craft-repair" ~ 1,
                                       adult_test$occupation == " Other-service" ~ 2,
                                       adult_test$occupation == " Sales" ~ 3,
                                       adult_test$occupation == " Exec-managerial" ~ 4,
                                       adult_test$occupation == " Prof-specialty" ~ 5,
                                       adult_test$occupation == " Handlers-cleaners" ~ 6,
                                       adult_test$occupation == " Machine-op-inspct" ~ 7,
                                       adult_test$occupation == " Adm-clerical" ~ 8,
                                       adult_test$occupation == " Farming-fishing" ~ 9,
                                       adult_test$occupation == " Transport-moving" ~ 10,
                                       adult_test$occupation == " Priv-house-serv" ~ 11,
                                       adult_test$occupation == " Protective-serv" ~ 12,
                                       adult_test$occupation == " Armed-Forces" ~ 13)
adult_test$num_occupation <- as.integer(adult_test$num_occupation)

adult_test$num_relationship <- case_when(adult_test$relationship == " Wife" ~ 0,
                                         adult_test$relationship == " Own-child" ~ 1,
                                         adult_test$relationship == " Husband" ~ 2,
                                         adult_test$relationship == " Not-in-family" ~ 3,
                                         adult_test$relationship == " Other-relative" ~ 4,
                                         adult_test$relationship == " Unmarried" ~ 5)
adult_test$num_relationship <- as.integer(adult_test$num_relationship)

adult_test$num_race <- case_when(adult_test$race == " White" ~ 0,
                                 adult_test$race == " Asian-Pac-Islander" ~ 1,
                                 adult_test$race == " Amer-Indian-Eskimo" ~ 2,
                                 adult_test$race == " Other" ~ 3,
                                 adult_test$race == " Black" ~ 4)
adult_test$num_race <- as.integer(adult_test$num_race)

adult_test$num_sex <- case_when(adult_test$sex == " Female" ~ 0,
                                adult_test$sex == " Male" ~ 1)
adult_test$num_sex <- as.integer(adult_test$num_sex)

adult_test$binary_income <- case_when(adult_test$income == " >50K." ~ 0,
                                      adult_test$income == " <=50K." ~ 1)
adult_test$binary_income <- as.factor(adult_test$binary_income)

# Removal of rows with missing data from the testing dataset
adult_test <- adult_test %>% filter(adult_test$workclass != " ?")
adult_test <- adult_test %>% filter(adult_test$native_country != " ?")
adult_test <- adult_test %>% filter(adult_test$occupation != " ?")

# Removal of rows with missing data from the training dataset
adult <- adult %>% filter(adult$native_country != " ?")
adult <- adult %>% filter(adult$workclass != " ?")
adult <- adult %>% filter(adult$occupation != " ?")

# Formatting of training dataset into useable form for gbm
adult_gbm <- adult[, c("age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week", "num_workclass", "num_marital_status", "num_occupation", "num_relationship", "num_race", "num_sex", "workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "income", "binary_income")]

adult$binary_income <- as.factor(adult$binary_income)

# Setting seed for random forest
set.seed(100)

# Construction of decision tree
adult_dt <- rpart(binary_income ~ age + fnlwgt + education_num + capital_gain + capital_loss + hours_per_week + num_workclass + num_marital_status + num_occupation + num_relationship + num_race + num_sex, data = adult)

# Prediction using decision tree
adult_dtpred <- predict(adult_dt, newdata = adult_test, type = "class")

# Construction of decision tree confusion matrix
conf_matrix_dt <- table(adult_dtpred, adult_test$binary_income)
confusionMatrix(conf_matrix_dt)

# Construction of random forest
adult_rf <- randomForest(binary_income ~ age + fnlwgt + education_num + capital_gain + capital_loss + hours_per_week + num_workclass + num_marital_status + num_occupation + num_relationship + num_race + num_sex, data = adult, importance = TRUE, ntree = 1000)

# Prediction using random forest
adult_rfpred <- predict(adult_rf, adult_test)

# Construction of random forest confusion matrix
conf_matrix_rf <- table(adult_rfpred, adult_test$binary_income)
confusionMatrix(conf_matrix_rf)

# Creation of gbm
adult.tc5.lr01 <- gbm.step(data = adult_gbm, gbm.x = 1:12, gbm.y = 21, family = "bernoulli", tree.complexity = 5, learning.rate = 0.01, bag.fraction = 0.5)

# Prediction using gbm
adult_gbmpred <- predict.gbm(adult.tc5.lr01, adult_test, n.trees = adult.tc5.lr01$gbm.call$best.trees, type = "response")

# Setting of prediction limit
pred.limit <- 0.5

# Construction of gbm confusion matrix
confusionMatrix(table(as.numeric(adult_gbmpred>pred.limit),adult_test$binary_income))