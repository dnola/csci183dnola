library('jpeg')
library('EBImage')
library('glmnet')
library('pROC')

filenames <- list.files("images_training_rev1", pattern="*.jpg", full.names=TRUE)

data = data.frame(read.csv('galaxy_train.csv'))
head(data)
#print(filenames)

N = length(filenames)
images = array(NA, dim=c(N,50,50))
classes = array(NA, dim=c(N))
ids = array(NA, dim=c(N))


plot(1:2, type='n')

for(i in 1:N) {
  name = filenames[i]
  id = as.integer(strsplit(strsplit(name,"/")[[1]][2],".jpg"))
  class = as.double(data[data$GalaxyID==id,]['Prob_Smooth'])
  classes[i] = class
  ids[i] = id
  img = readJPEG(name)
  img = resize(img,50,50)
  #img = getValues(img)
  bw = img[,,1]*0.21 + img[,,2]*0.72 + img[,,3]*0.07
  if (i %% 100==0) { print(i)}
  if (i %% 1000==1) { rasterImage(bw, 1, 1, 2, 2) }
  images[i,,] = bw
}

features = array(NA, dim=c(N,6))

for(i in 1:N){
  cur = as.vector(images[i,,])
  features[i,] = c(mean(cur),var(cur),quantile(cur,.10),quantile(cur,.25),quantile(cur,.75),quantile(cur,.90))
}

print(features)
f = as.data.frame(features)
f$prob = classes
rownames(f) <- ids

train = f[apply(!is.na(f$prob), 1, any),]
L = nrow(train)
print(L)

cv = train[(as.integer(L/2)+1):L,]

train = train[1:as.integer(L/2),]
test = f[apply(is.na(f$prob), 1, any),]

head(train)
head(test)
head(cv)

selected = c('V1', 'V2','V3', 'V4', 'V5', 'V6')
#selected = c('V6', 'V4')
train_mat = as.matrix(train[selected])

gl = cv.glmnet(train_mat,as.vector(train$prob))
cv_mat = as.matrix(cv[selected])
preds = predict(gl,cv_mat,s=gl$lambda.min)


error = (length(preds) - sum(sqrt((preds - as.vector(cv$prob))**2)))/length(preds)

print("RMSE:")
print(error)

