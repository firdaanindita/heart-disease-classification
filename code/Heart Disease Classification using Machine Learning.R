library(caret)
library(pscl)
library(ResourceSelection)
library(randomForest)

#load dataset
heart=read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",header=FALSE,sep=",",na.strings = '?')
names(heart) = c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg","thalach","exang", "oldpeak","slope", "ca", "thal", "target")
heart$target=ifelse(heart$target==0,0,1)
heart$sex=as.factor(heart$sex)
heart$cp=as.factor(heart$cp)
heart$fbs=as.factor(heart$fbs)
heart$restecg=as.factor(heart$restecg)
heart$exang=as.factor(heart$exang)
heart$slope=as.factor(heart$slope)
heart$thal=as.factor(heart$thal)
heart$target=-as.factor(heart$target)
str(heart)

#remove missing value
dataset=na.omit(heart)
str(dataset)

#randomize
set.seed(42)
rows = sample(nrow(dataset))
dataset = dataset[rows, ]

#Hold Out
train.flag = createDataPartition(y=dataset$target, p=0.8, list=FALSE)
training = dataset[ train.flag, ]
testing = dataset[-train.flag, ]

#loren
logit1=glm(family=binomial,data=training,target~.)
summary(logit1)
pR2(logit1)
qchisq(0.95,13)

logit2 =glm(family=binomial,data=training,target~sex+cp+oldpeak+ca+thal)
summary(logit2)
pR2(logit2)
qchisq(0.95,5)
hoslem.test(logit2$y, fitted(logit2), g=10)

model1 = train(target~.,  data=training, method="glm", family=binomial(link="logit"))
summary(model1) =
model2 = train(target~sex+cp+oldpeak+ca+thal,  data=training, method="glm",family=binomial(link="logit"))
summary(model2)

train_pred = predict(model2, newdata=training)
predicttrain = predict(model2, training, type="prob") 
CMtrain = confusionMatrix(train_pred, training[,"target"])
CMtrain 

test_pred = predict(model2, newdata=testing)
predicttest = predict(model2, testing, type="prob") 
CMtest = confusionMatrix(test_pred, testing[,"target"])
CMtest

#random forest
x = dataset[,1:13]
y=subset(dataset, select = c(14))

customRF = list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters = data.frame(parameter = c("mtry", "ntree", "nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree", "nodesize"))
customRF$grid = function(x, y, len = NULL, search = "grid") {}
customRF$fit = function(x, y, wts, param, lev, last, weights, classProbs, ...) 
{randomForest(x, y, mtry = param$mtry, ntree=param$ntree, nodesize=param$nodesize, ...)}
customRF$predict = function(modelFit, newdata, preProc = NULL, submodels = NULL)predict(modelFit, newdata)
customRF$prob = function(modelFit, newdata, preProc = NULL, submodels = NULL)predict(modelFit, newdata, type = "prob")
customRF$sort = function(x) x[order(x[,1]),]
customRF$levels = function(x) x$classes
fitControl = trainControl(method = "repeatedcv",  number = 3,   repeats = 10)
grid = expand.grid(.mtry=c(2, 3, 4,5,6), 
                    .ntree = c(100,200, 300,400, 500,600,700,800,900, 1000),
                    .nodesize =c(1:5))
set.seed(123)
fit_rf = train(target ~ ., data = training, method = customRF, tuneGrid= grid,trControl = fitControl)

fit_rf$finalModel
plot(fit_rf)

pred_train = predict(fit_rf, training)
confusionMatrix(pred_train,training[,14]) 

pred_test = predict(fit_rf, testing)
confusionMatrix(pred_test,testing[,14]) 
