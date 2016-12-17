##### Gensim_analysis.R
##### Creates Gensim Analysis Files, where we verify whether the gensim_model correctly grouped/classified (seed, response) pairs based on common misconceptions.


# *************************************
# PART I. Reading in Data
# *************************************
require(reshape2)
require(dplyr)
wd <- "~/Documents/School/INFO 290/understandingMultiplying3"
setwd(wd)

#source("../frequency_results/analysis.R")

um.3 <- read.csv("../frequency_results/understandingMultiplying3.csv", stringsAsFactors = F)
um.3.analysis <- read.csv("../frequency_results/understandingMultiplying3_analysis.csv", stringsAsFactors = F)

first.5.3 <- 
  um.3 %>% 
  group_by(seed) %>% 
  arrange(desc(frequency)) %>% 
  slice(c(1:5))

#Doing this to get correct responses in the end (Excel formats it incorrectly)
first.5.3$explanation <- um.3.analysis$explanation


# *************************************
# PART II. Exporting Data
# *************************************

#Top 3 types with most gensim outputs
types <- c("Type_2.csv", "Type_2_SOLVE_AREA.csv","Type_3_TAPE_DIAGRAMS.csv")

#Not explanation/issues we want from the main gensim response because these do not explain much about misconceptions
toMatch <- c("correct", "Correct", "no idea", "typo", "Typo", "Entering space to get hint", "entering space to get hint", " ", "", "Not follow directions, but correct answer", "not follow directions")
rel.first.5.3 <- first.5.3[!first.5.3$explanation %in% toMatch, ]


#Exporting gensim analysis files based on type, including explanation for the main gensim (seed,response) tuple.
for(i in types){
  rawdata <- read.csv(i, stringsAsFactors = F)
  relevant <- rawdata[rawdata$rank <= 5, ]
  relevant <- inner_join(relevant, rel.first.5.3[, -2], by = c("seed", "response"))
  sample_rows <- sample(1:nrow(relevant), 5)
  small <- relevant[sample_rows, ]
  write.csv(small, paste0(i, "_analysis.csv"))
}




# all_seeds <- list()
# 
# 
# 
# for(i in 1:nrow(relevant)){
#   rawstring <- relevant[i, 1]
#   #Clean square brackets
#   cleaned_string <- gsub("\\[|\\]", "", rawstring)
#   
#   #Split by end of tuple for each seed
#   cleaned_string <- unlist(strsplit(cleaned_string, "), "))
#   
#   #Remove everything after the , (the answers)
#   cleaned_string <- gsub(',.*', "", cleaned_string)
#   
#   #Remove more characters just to get the seed
#   seed <- gsub("[(']", "", cleaned_string)
#   all_seeds[[i]] <- seed[1:5]
# }
# 
# names(all_seeds) <- relevant$seed
