library('choroplethr')
library('acs')

api.key.install(key="af813de7c712487dd60b8a148a336480701d60fd")
dat = get_acs_data("B08101", "county",column_idx = 41)
df = as.data.frame(dat)

max = max(df$'df.value')

df$rate = df$'df.value'/max
df$id = df$'df.region'
head(df)

clean = subset(df, select=c('id','rate'))
rownames(clean) <- clean$id
head(clean)

write.table(clean, file='bike_country.tsv', quote=FALSE, sep='\t')

