dat = read.csv("bike.csv", header = TRUE)

nrow(dat)

head(dat)
print(dat$LOCATION)
dat$F = 1

dat$LOC = "Other"
for(i in c("Swig", "Casa", "Sobrato", "Benson", "Bellarmine", "Sanfilippo", "Graham", "Domicilio", "Dunne", "Campisi", "Walsh", "Villas","Nobili", "McLaughlin", "Loyola", "St. Clare"))
{
  dat[grep(i, dat$LOCATION),]$LOC = i
}

print(dat[grep("Other",dat$LOC),]$LOCATION)

print(dat$DATE)


worst_dorms = tapply(dat$F, dat$LOC, FUN=sum)

worst_dorms = data.frame(worst_dorms )

print( worst_dorms[order(worst_dorms),])

dat$CLEAN = dat$DATE
head(dat$CLEAN)
clean_days = dat[grep("-", dat$DATE, invert=TRUE),]

print(clean_days)
#install.packages('lubridate')
library(lubridate) 

clean_days$DOW <- wday(mdy(clean_days$DATE), label=TRUE)

print(clean_days)

worst_days = tapply(clean_days$F, clean_days$DOW, FUN=sum)
worst_days = data.frame(worst_days )
print( worst_days[order(-worst_days),])
barplot(worst_days$worst_days)

clean_days$MOY <- month(mdy(clean_days$DATE), label=TRUE)

print(clean_days)

worst_months = tapply(clean_days$F, clean_days$MOY, FUN=sum)
worst_months = data.frame(worst_months )
print( worst_months[order(-worst_months),])
barplot(worst_months$worst_months)



