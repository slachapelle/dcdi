# Copyright (c) 2018-2020, Joris M. Mooij. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

gaussCIcontexttest_slow <- function(x, y, S, suffStat) {
  stopifnot(dim(data)[1] >= 1)

  data<-suffStat$data
  p<-dim(data)[2]
  contextVars<-suffStat$contextVars
  verbose<-suffStat$verbose

  if( verbose ) 
    cat( 'Testing ', x, '_||_', y, '|', S, '\n' )
  stopifnot(length(x) == 1 && length(y) == 1)
  stopifnot(length(intersect(x,y)) == 0)
  stopifnot(length(intersect(union(x,y),S)) == 0)

  ScV <- intersect(S,contextVars)
  if( is.null(ScV) || length(ScV)==0 ) {
    pval <- parcortest(x,y,S,data[,],removeNAs=suffStat$removeNAs,verbose=FALSE)
  } else {
    uniqueScVvals<-as.matrix(unique(data[,ScV]))
    pvals<-c()
    nRegimes<-dim(uniqueScVvals)[1]
    for( R in 1:nRegimes ) {
      # let inds_R be the row indices for the R'th unique value of the context vars in S 
#      inds_R<-1:dim(data)[1]
#      for( i in 1:length(ScV) )
#        inds_R<-intersect(inds_R,which(data[,ScV[i]]==uniqueScVvals[R,i]))
      
      inds_R<-1:dim(data)[1]
      for( i in 1:length(ScV) )
        inds_R<-inds_R[which(data[inds_R,ScV[i]]==uniqueScVvals[R,i])]

      if( verbose )
        cat('R=',R,'/',nRegimes,'(',uniqueScVvals[R,],'): ')

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
      pval<-pchisq(stat, df=2*length(pvals), lower.tail=FALSE)
      if( verbose )
        cat('pvals=',pvals,': ',stat,'->',pval,'\n')
    } else {
      pval<-NA
      if( verbose )
        cat('pvals=()->NA\n')
    }
  }
  pval
}
