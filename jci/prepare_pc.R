# Copyright (c) 2018-2020, Joris M. Mooij. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

prepare_pc <- function(data,systemVars,contextVars,subsamplefrac=0.0,mode='merge',obsContext=matrix(0,1,length(contextVars))) {
  # data:          Nxp matrix
  # systemVars:    indices of system variables (in 1..p)
  # contextVars:   indices of context variables (in 1..p)
  # subsamplefrac: fraction of subsamples used for subsampling; if 0, don't subsample and use all data
  # mode:          one of {'multiple','obsonly','merge'}, default='merge'
  # obsContext:    vector of values of context variables that defines the observational context; 
  #                  if nonempty, use only data from the observational context,
  #                  if empty or NULL, use all data
  #
  # returns result list, with fields {'data','systemVars','contextVars','N','p','removeNAs'}

  suppressMessages(library(mgcv))

  # check if this is sensible
  stopifnot( length(systemVars) > 0 )
  stopifnot( mode %in% c('multiple','obsonly','merge') )

  N<-dim(data)[1]
  stopifnot( N > 0 )
  labels<-colnames(data)
  if( mode=='obsonly' && length(contextVars) > 0 ) { # if mode=='obs', throw away non-observational data
    stopifnot( !is.null(obsContext) )
    inds = c()
    for( i in 1:N ) {
      if( norm(as.matrix(data[i,contextVars] - obsContext)) == 0 )
        inds = c(inds,i)
    }
    data<-data[inds,]
    N<-dim(data)[1]
    stopifnot( N > 0 )
  } else if( mode=='merge' && length(contextVars) > 0 ) {
    # find indices for unique joint values of context variables
    uniqueContextValues<-uniquecombs(data[,contextVars])
    regimes<-attr(uniqueContextValues,'index')
  }
  
  # subsample rows, if required
  if( subsamplefrac == 0.0 ) {
    rows<-1:N
  } else {
    rows<-sample(N,round(subsamplefrac*N),replace=TRUE)
  }
  stopifnot( length(rows) > 0 )

  # select subset of columns corresponding to system variables and context variables and subset of (subsampled) rows
  if( length(contextVars) == 0 ) {
    data<-data[rows,systemVars]
    labels<-labels[systemVars]
  } else {
    if( mode=='merge' ) {
      data<-cbind(data[rows,systemVars],regimes[rows])
      labels<-c(labels[systemVars],'regime')
      contextVars<-c(length(systemVars)+1)
    } else {
      data<-cbind(data[rows,systemVars],data[rows,contextVars])
      labels<-c(labels[systemVars],labels[contextVars])
      contextVars<-(length(systemVars)+1):(length(systemVars)+length(contextVars))
    }
  }
  systemVars<-1:length(systemVars)
  colnames(data)<-labels
  N<-dim(data)[1]
  p<-dim(data)[2]

  # check for any remaining NAs in the data
  if( anyNA(data) )
    removeNAs <- TRUE
  else
    removeNAs <- FALSE

  # return result
  result = list(data=data,systemVars=systemVars,contextVars=contextVars,N=N,p=p,removeNAs=removeNAs)
  return(result)
}
