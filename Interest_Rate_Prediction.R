
## ---- Project: Predicting Loan Interest Rates via Multi-Model Analysis

# Objective: Develop a predictive model for interest rates based on borrower profiles

# Loading the dataset
ld=read.csv("loans data.csv",stringsAsFactors = FALSE)
library(dplyr)
glimpse(ld)

## ---- DATA PREPROCESSING & FEATURE ENGINEERING

# Cleaning numeric fields by removing special characters and casting types
ld=ld %>%
  mutate(Interest.Rate=as.numeric(gsub("%","",Interest.Rate)) ,
         Debt.To.Income.Ratio=as.numeric(gsub("%","",Debt.To.Income.Ratio)) ,
         Open.CREDIT.Lines=as.numeric(Open.CREDIT.Lines) , 
         Amount.Requested=as.numeric(Amount.Requested) ,
         Amount.Funded.By.Investors=as.numeric(Amount.Funded.By.Investors),
         Revolving.CREDIT.Balance=as.numeric(Revolving.CREDIT.Balance)
  )
glimpse(ld)

# Excluding 'Amount.Funded.By.Investors' (not known at the time of application)
ld = ld %>%
  select(-Amount.Funded.By.Investors)
glimpse(ld)

# Tranforming 'fico': Derived as the midpoint of the provided FICO Range
ld= ld %>%
  mutate(f1=as.numeric(substr(FICO.Range,1,3)),
         f2=as.numeric(substr(FICO.Range,5,7)),
         fico=0.5*(f1+f2)
  ) %>%
  select(-FICO.Range,-f1,-f2)
glimpse(ld)

table(ld$Employment.Length)

# Transforming 'Employment.Length' into a continuous numeric scale for modeling
ld=ld %>%
  mutate(el=ifelse(substr(Employment.Length,1,2)=="10",10,Employment.Length),
         el=ifelse(substr(Employment.Length,1,1)=="<",0,el),
         el=gsub("years","",el),
         el=gsub("year","",el),
         el=as.numeric(el)
  ) %>%
  select(-Employment.Length) %>%
  na.omit()

# One-Hot Encoding: Transforming 'Home.Ownership' into binary indicators
table(ld$Home.Ownership)

ld=ld %>%
  mutate(HW_RENT=as.numeric(Home.Ownership=="RENT"),
         HW_MORT=as.numeric(Home.Ownership=="MORTGAGE"),
         HW_OWN=as.numeric(Home.Ownership=="OWN")) %>%
  select(-Home.Ownership)

# Consolidating 'Loan.Purpose' into broader categories based on average interest rate impact
table(ld$Loan.Purpose)
round(tapply(ld$Interest.Rate,ld$Loan.Purpose,mean),0)

# 1 : car,educational, major_purchase - 11
# 2 : credit_card, house, small_business, other - 13
# 3 : home_improvement, medical, vacation, wedding - 12
# 4 : debt_consolidation, moving - 14
# 5 : renewable_energy - 10

ld=ld %>%
  mutate( LP_1=as.numeric(Loan.Purpose %in% c("car","educational", "major_purchase")),
          LP_2=as.numeric(Loan.Purpose %in% c("credit_card","house","small_business","other")),
          LP_3=as.numeric(Loan.Purpose %in% c("home_improvement"," medical", "vacation", "wedding")),
          LP_4=as.numeric(Loan.Purpose %in% c("debt_consolidation","moving"))
  ) %>%
  select(-Loan.Purpose)

# Creating an indicator for 36-month loan terms (Baseline: 60 months)
table(ld$Loan.Length)

ld=ld %>%
  mutate(LL_36=as.numeric(Loan.Length=="36 months")) %>%
  select(-Loan.Length)

# # Target Encoding: Grouping states into 'Tiers' based on observed mean interest rates
# This reduces the dimensionality of 'State' 
# preventing model overfitting and addressing the high cardinality of geographic data
table(ld$State)
round(tapply(ld$Interest.Rate, ld$State, mean), 0)

# 1.Low_int_rates (<=12) - "SD", "MT", "DE", "KY","MA","NH", "OH"
# 2.High_int_rates (>=16) - "HI", "AK", "MS", "VT"
# 3.Medium_int_rates (13-15) - Remaining States

ld <- ld %>%
  mutate(State_Tier = case_when(
    State %in% c("SD", "MT", "DE", "KY","MA","NH", "OH") ~ "Low_Rate_State",
    State %in% c("HI", "AK", "MS", "VT") ~ "High_Rate_State",
    TRUE ~ "Mid_Rate_State" # Baseline category representing the majority of observations
  ))

ld <- ld %>%
  mutate(
    State_High_Risk = as.numeric(State_Tier == "High_Rate_State"),
    State_Low_Risk  = as.numeric(State_Tier == "Low_Rate_State")
  ) %>%
  select(-State, -State_Tier)

glimpse(ld)

# Checking columns for number of NA
apply(ld,2,function(x) sum(is.na(x)))

## ---- MODELING & VALIDATION

# 70/30 Train-Test Split to evaluate model performance
set.seed(2)
s=sample(1:nrow(ld),0.70*nrow(ld))
ld_train=ld[s,]
ld_test=ld[-s,]

glimpse(ld_train)

# Computing correlation between different variables
View(cor(ld_train))

## 1. LINEAR REGRESSION 
# Initial Fit: Including all features except ID
fit= lm(Interest.Rate ~ .-ID ,data=ld_train)
summary(fit)

# Multicollinearity Check: Using Variance Inflation Factor (VIF) to detect redundant features
library(car)
t=vif(fit)
sort(t,decreasing = TRUE)

# Refining model by iteratively removing high-VIF variables
fit=lm(Interest.Rate~. -ID - HW_MORT,data=ld_train)
t=vif(fit)
sort(t,decreasing = TRUE)

fit=lm(Interest.Rate~. -ID - HW_MORT-LP_4,data=ld_train)
t=vif(fit)
sort(t,decreasing = TRUE)

summary(fit)

# Stepwise Regression by removing non-significant variables (p > 0.05)
fit=lm(Interest.Rate~. -ID - HW_MORT-LP_4-State_Low_Risk,data=ld_train)
summary(fit)

fit=lm(Interest.Rate~. -ID - HW_MORT-LP_4-State_Low_Risk-Monthly.Income,data=ld_train)
summary(fit)

fit=lm(Interest.Rate~. -ID - HW_MORT-LP_4-State_Low_Risk-Monthly.Income-LP_3,data=ld_train)
summary(fit)

fit=lm(Interest.Rate~. -ID - HW_MORT-LP_4-State_Low_Risk-Monthly.Income-LP_3-Debt.To.Income.Ratio,data=ld_train)
summary(fit)

fit=lm(Interest.Rate~. -ID - HW_MORT-LP_4-State_Low_Risk-Monthly.Income-LP_3-Debt.To.Income.Ratio-el,data=ld_train)
summary(fit)

fit=lm(Interest.Rate~. -ID - HW_MORT-LP_4-State_Low_Risk-Monthly.Income-LP_3-Debt.To.Income.Ratio-el-LP_2,data=ld_train)
summary(fit)

fit=lm(Interest.Rate~. -ID - HW_MORT-LP_4-State_Low_Risk-Monthly.Income-LP_3-Debt.To.Income.Ratio-el-LP_2-LP_1,data=ld_train)
summary(fit)

#Final model
fit_train=lm(Interest.Rate~. -ID - HW_MORT-LP_4-State_Low_Risk-Monthly.Income-LP_3-Debt.To.Income.Ratio-el-LP_2-LP_1,data=ld_train)
summary(fit)

# Model : Pred_Interest_Rate =74.78 + Amount.Requested*1.473*10^-4 + Open.Credit.Lines*(-3.238*10^-2) + Revolving.Credit.Balance*(-5.596*10^-6) + Inquiries.in.the.Last.6.Months*3.745*10^-1 + fico*(-8.634*10^-2) + HW_RENT*2.874*10^-1 + HW_OWN*5.316*10^-1 + LL_36*(-3.293) + State_High_Risk*1.366
    
train_res = cbind.data.frame(Actual=ld_train$Interest.Rate, Fitted=fitted(fit_train), Error=residuals(fit_train))
View(train_res)

rmse_train=sqrt(mean(train_res$Error^2))
rmse_train

# Testing Linear Regression Assumptions on train data

# Actual vs predicted
library(ggplot2)
ggplot(train_res,aes(x=Actual,y=Fitted))+geom_point()

# Error~N(0,sigma2)
ggplot(train_res,aes(Error))+geom_histogram()

# Predicted vs error - Homoscedasticity/Independence of errors
ggplot(train_res,aes(x=Fitted,y=Error))+geom_point()

# Prediction on the Test data
ir_predict=predict(fit_train,newdata=ld_test)
TestRes = cbind.data.frame(Act=ld_test$Interest.Rate, Pred=ir_predict)
View(TestRes)

plot(ld_test$Interest.Rate,ir_predict)

res=ld_test$Interest.Rate-ir_predict

# Evaluating Performance (RMSE)
rmse_test=sqrt(mean(res^2))
rmse_test

# Testing Linear Regression Assumptions on test data

d=data.frame(real=ld_test$Interest.Rate,predicted=ir_predict, Res=ld_test$Interest.Rate-ir_predict)

# Actual vs predicted
ggplot(d,aes(x=real,y=predicted))+geom_point()

# Error~N(0,sigma2)
ggplot(d,aes(Res))+geom_histogram()

# Predicted vs error - Homoscedasticity/Independence of errors
ggplot(d,aes(x=predicted,y=Res))+geom_point()

## 2. DECISION TREE

library(rpart)
library(rpart.plot)

# Fitting the Decision tree model 

tree_fit = rpart(Interest.Rate ~ . -ID, data = ld_train)

prp(tree_fit, 
    type = 2, 
    extra = 1, 
    main = "Decision Tree for Interest Rate")

# Prediction on the Test data
tree_pred = predict(tree_fit, newdata = ld_test)

# Evaluating Performance (RMSE)
tree_res = ld_test$Interest.Rate - tree_pred
rmse_tree = sqrt(mean(tree_res^2))

print(paste("Decision Tree RMSE:", rmse_tree))

# Visualizing Actual vs Predicted for Tree
d_tree = data.frame(real = ld_test$Interest.Rate, predicted = tree_pred)
ggplot(d_tree, aes(x = real, y = predicted)) + 
  geom_point(color = "blue") +
  geom_abline(color = "red") +
  ggtitle("Decision Tree: Actual vs Predicted")


## 3. RANDOM FOREST
library(randomForest)

# Fitting the Random Forest model

set.seed(2) 
rf_fit = randomForest(Interest.Rate ~ . -ID, 
                      data = ld_train, 
                      ntree = 500,
                      importance = TRUE)

print(rf_fit)

# Prediction on the Test Data
rf_pred = predict(rf_fit, newdata = ld_test)

# Evaluating Performance (RMSE)
rf_res = ld_test$Interest.Rate - rf_pred
rmse_rf = sqrt(mean(rf_res^2))

print(paste("Random Forest RMSE:", rmse_rf))

# Variable Importance Plot
# This shows which variables were most useful in predicting Interest Rate
varImpPlot(rf_fit, main = "Random Forest Variable Importance")

## ------------------------------------------------------------------------
## FINAL MODEL COMPARISON
## ------------------------------------------------------------------------

# Calculating Test RMSE (Root Mean Squared Error) to determine predictive accuracy

comparison = data.frame(
  Model = c("Linear Regression", "Decision Tree", "Random Forest"),
  RMSE = c(rmse_test, rmse_tree, rmse_rf)
)

print(comparison)

# Selecting the model with the lowest error
best_model = comparison[which.min(comparison$RMSE), ]
print(paste("The best model is", best_model$Model, "with RMSE:", round(best_model$RMSE, 4)))