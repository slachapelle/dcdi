# Copyright (c) 2018-2020, Joris M. Mooij. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

gaussCIcontexttest <- function(x, y, S, suffStat) {
  data<-suffStat$data
  p<-dim(data)[2]
  contextVars<-suffStat$contextVars
  verbose<-suffStat$verbose
  uniqueContextValues<-suffStat$uniqueContextValues
  regimes<-suffStat$regimes

  if( verbose ) 
    cat( 'Testing ', x, '_||_', y, '|', S, '\n' )
  stopifnot(length(x) == 1 && length(y) == 1)
  stopifnot(length(intersect(x,y)) == 0)
  stopifnot(length(intersect(union(x,y),S)) == 0)

  #S <- union(S, setdiff(contextVars, union(x,y)))  # JORIS

  ScV <- intersect(S,contextVars)
  skiptests <- FALSE
  if( 'skiptests' %in% names(suffStat) )
    skiptests <- suffStat$skiptests
  if( skiptests && length(setdiff(contextVars,union(c(x,y),S))) > 0 ) {
    pval <- 0
  } else {
    if( is.null(ScV) || length(ScV)==0 ) {
      pval <- parcortest(x,y,S,data[,],removeNAs=suffStat$removeNAs,verbose=FALSE)
    } else {
      # find indices of ScV in contextVars
      ScVindices<-match(ScV,contextVars)
      # find unique values of context variables in ScV
      uniqueScVcombs<-uniquecombs(uniqueContextValues[,ScVindices])
      pvals<-c()
      nRegimes<-dim(uniqueScVcombs)[1]
      for( R in 1:nRegimes ) {
        # get all regimes with this unique value of context variables in ScV
        setOfRegimes<-which(attr(uniqueScVcombs,'index') == R)
        # let inds_R be the row indices for the R'th unique value of the context vars in S 
        inds_R<-which(regimes %in% setOfRegimes)

        if( verbose )
          cat('R=',R,'/',nRegimes,'(',as.matrix(uniqueScVcombs)[R,],'): ')

        Sremain<-setdiff(S,ScV)
        p <- parcortest(x,y,Sremain,data[inds_R,],removeNAs=suffStat$removeNAs,verbose=FALSE)
        if( !is.na(p) ) {
          pvals<-c(pvals,p)
          if( verbose )
            cat(p,';','\n')
        } else {
          if( verbose )
            cat('skipping this regime\n')
        }
      }

      if( length(pvals) >= 1 ) {
        # Fisher's method (should we replace by min over p-values?)
        stat<-(-2.0) * sum(log(pvals))
        # distributed as chi^2 with 2*N d.o.f.
        pval<-pchisq(stat, df=2*length(which(pvals<1)), lower.tail=FALSE)
        if( verbose )
          cat('pvals=',pvals,': ',stat,'->',pval,'\n')
      } else {
        pval<-NA
        if( verbose )
          cat('pvals=()->NA\n')
      }
    }
  }
  pval
}
