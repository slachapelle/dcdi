# Copyright (c) 2018-2020, Joris M. Mooij. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

parcortest <- function(x, y, S, data, removeNAs=TRUE, verbose=FALSE) {
  # standard partial correlation test
  suppressMessages(library(ppcor))

  if( verbose ) 
    cat( 'Testing ', x, '_||_', y, '|', S, '\n' )
  stopifnot(length(x) == 1 && length(y) == 1)
  stopifnot(length(intersect(x,y)) == 0)
  stopifnot(length(intersect(union(x,y),S)) == 0)

  if( removeNAs ) {
    inds <- which(apply(data[,c(x,y,S)],1,anyNA)==FALSE) # remove NAs
    if( verbose )
      cat('Removed NAs: went from',dim(data)[1],'to',length(inds),'samples.\n')

    if( is.null(S) || length(S)==0 ) {
      if( length(unique(data[inds,x])) > 1 && length(unique(data[inds,y])) > 1 && length(inds) > 2 ) {
        pval <- cor.test(data[inds,x],data[inds,y])$p.value
      } else {
        pval <- NA
      }
    } else {
      if( length(unique(data[inds,x])) > 1 && length(unique(data[inds,y])) > 1 && length(inds) > 2 ) {
        pval <- pcor.test(data[inds,x],data[inds,y],data[inds,S])$p
      } else {
        pval <- NA
      }
    }
  } else {
    if( is.null(S) || length(S)==0 ) {
      pval <- NA
      pval <- tryCatch({
        cor.test(data[,x],data[,y])$p.value
      }, warning=function(w) {
        if( length(unique(data[,x])) == 1 || length(unique(data[,y])) == 1 ) {
          return(1)
        } else {
          stop('This has to be investigated')
        }
      })
    } else {
      pval <- NA
      pval <- tryCatch({
        pcor.test(data[,x],data[,y],data[,S])$p
      }, warning=function(w) {
        if( length(unique(data[,x])) == 1 || length(unique(data[,y])) == 1 ) {
          return(1)
        } else {
          stop('This has to be investigated')
        }
      }) 
    }
  }

  pval
}
