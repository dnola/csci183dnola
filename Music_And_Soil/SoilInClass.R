install.packages('glmnet')
library(glmnet)

train = data.frame(read.csv('soil.csv'))
head(train)

classes = train$pH
print(classes)
mat = as.matrix(train[,0:(ncol(train)-1)])

gl =cv.glmnet(mat,classes, family="gaussian")
plot(gl$glmnet.fit)

cos = as.matrix(coef(gl, lambda=gl$lambda.min))

cos_nz = cos[cos>0,]
head(cos_nz)

plot(cos)
plot(cos_nz)

top = tail(sort(cos_nz),11) # Prints intercept too, so grab an extra one
print(top)
