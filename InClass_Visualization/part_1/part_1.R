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

head(dat)



worst_dorms = tapply(dat$F, dat$LOC, FUN=sum)

worst_dorms = data.frame(worst_dorms )

print( worst_dorms[order(worst_dorms),])

dat$CLEAN = dat$DATE
head(dat$CLEAN)

tail(dat)

library(lubridate) 
clean_days = dat[grep("-", dat$DATE, invert=TRUE),]
clean_days$MOY = month(mdy(clean_days$DATE), label=TRUE)
head(clean_days)


head(clean_days,300)

print(length(clean_days$MOY))

clean_days$ymd = ymd(clean_days$DATE)
clean_days$format = format(mdy(clean_days$DATE), format="%Y-%m")
head(clean_days)

by_month_dorm = aggregate(clean_days$F ~ clean_days$LOC+clean_days$format, FUN = sum)

head(by_month_dorm)
by_month_dorm$date = by_month_dorm$'clean_days$format'
by_month_dorm$count = by_month_dorm$'clean_days$F'
by_month_dorm$name = by_month_dorm$'clean_days$LOC'

library('tidyr')
data_wide <- spread(subset(by_month_dorm, select=c('date','count','name')), date, count)


data_wide[is.na(data_wide)] <- 0
head(data_wide)

write.table(data_wide, file='bike_scu.tsv', quote=FALSE, sep='\t',  row.names = F)

