setwd("[pathway]")

# load the file
obj_name <- load("ADNIMERGE2/data/ADSL.rda")

print(obj_name)   # shows object name

# extract object
df <- get(obj_name)

# write CSV
write.csv(df, "ADSL.csv", row.names = FALSE)

# confirm it exists
print(list.files())

setwd("[pathway]")

# load the file
obj_name <- load("ADNIMERGE2/data/UCSFFSX6.rda")

print(obj_name)   # shows object name

# extract object
df <- get(obj_name)

# write CSV
write.csv(df, "UCSFFSX6.csv", row.names = FALSE)

# confirm it exists
print(list.files())
