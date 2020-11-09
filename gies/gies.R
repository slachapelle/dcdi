# MIT License
#
# Copyright (c) 2018 Diviyan Kalainathan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

library(methods)
library(pcalg)

set.seed(42)

dataset <- read.csv(file='{FOLDER}{FILE}', sep=",", header=FALSE);

if({SKELETON}){
  fixedGaps <- read.csv(file='{FOLDER}{GAPS}', sep=",", header=FALSE) # NULL
  fixedGaps = (data.matrix(fixedGaps))
  rownames(fixedGaps) <- colnames(fixedGaps)
}else{
  fixedGaps = NULL
}

if({INTERVENTION}){
  target_raw <- read.csv(file='{FOLDER}{TARGETS}', sep=",", header=FALSE)
  target_raw <- target_raw + 1
  
  targets = list()
  # getting the unique rows for the targets
  uniq = unique(target_raw)
  # creating the targets, which is a unique value list
  for(i in 1:dim(uniq)[1]){
    targets[[i]] = as.integer(uniq[i,!is.na(uniq[i,])])
  }
  
  # creating the targets index, which is of same length as a and points to unique values that are used
  target.index = c()
  
  # function that will return the index of the unique vector from the matrix. 
  get_index<-function(matrix, vector){
    for(i in 1:length(matrix)){
      if (length(vector) == length(matrix[i][[1]])){
        if(all(vector %in% matrix[i][[1]])){
          return (i)
        }
      }
    }
  }
  
  # getting the target.index for each row of a
  for (i in 1:dim(target_raw)[1]){
    target.index = c(target.index, get_index(targets, target_raw[i,][!is.na(target_raw[i,])]))
  }

  score <- new("{SCORE}", data=dataset, targets=targets, target.index=target.index, lambda = {LAMBDA})
  result <- pcalg::gies(score, fixedGaps=fixedGaps, targets=score$getTargets())
}else{
  score <- new("{SCORE}", data = dataset, lambda = {LAMBDA})
  result <- pcalg::gies(score, fixedGaps=fixedGaps)
}

gesmat <- as(result$repr, "matrix") #repr essgraph
gesmat[gesmat] <- 1
  #gesmat[!gesmat] <- 0
write.csv(gesmat, row.names=FALSE, file = '{FOLDER}{OUTPUT}');
