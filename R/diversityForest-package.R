##' Diversity Forests
##' 
##' The diversity forest algorithm is not a specific algorithm, but an alternative candidate split sampling scheme 
##' that makes complex split procedures in random forests possible computationally by drastically reducing 
##' the numbers of candidate splits that need to be evaluated for each split. It also avoids the well-known variable 
##' selection bias in conventional random forests that has the effect that variables with many possible splits 
##' are selected too frequently for splitting (Strobl et al., 2007). For details, see Hornung (2020).
##'
##' This package currently features two types of diversity forests:
##' \itemize{
##' \item the basic form of diversity forests that uses univariable, binary splitting, which is also used
##' in conventional random forests
##' \item interaction forests (IFs) (Hornung & Boulesteix, 2021), which use bivariable splitting to model quantitative and qualitative interaction effects.
##' IFs feature the Effect Importance Measure (EIM), which ranks the variable pairs with respect to the predictive importance
##' of their quantitative and qualitative interaction effects. The individual variables can be ranked as well
##' using EIM. For details, see Hornung & Boulesteix (2021).
##' }
##' Diversity forests with univariable splitting can be constructed using the function \code{\link{divfor}} and 
##' interaction forests using the function \code{\link{interactionfor}}. Both functions support categorical,
##' metric, and survival outcomes.
##'
##' This package is a fork of the R package 'ranger' that implements random forests using an
##' efficient C++ implementation. The documentation is in large parts taken from
##' 'ranger', where some parts of the documentation may not apply to (the current version of) the 'diversityForest' package.
##'
##' Details on further functionalities of the code that are not presented in the help pages of 'diversityForest' are found
##' in the help pages of 'ranger', version 0.11.0, because 'diversityForest' is based on the latter version of 'ranger'. 
##' The code in the example sections can be used as a template for all basic application scenarios with respect to classification, 
##' regression and survival prediction.
##'
##' @references
##' \itemize{
##'   \item Hornung, R. (2020) Diversity Forests: Using split sampling to allow for complex split procedures in random forest. Technical Report No. 234, Department of Statistics, University of Munich. \url{https://epub.ub.uni-muenchen.de/73377/index.html}.
##'   \item Hornung, R. & Boulesteix, A.-L. (2021) Interaction Forests: Identifying and exploiting interpretable quantitative and qualitative interaction effects. Technical Report No. 237, Department of Statistics, University of Munich. \url{https://epub.ub.uni-muenchen.de/75432/index.html}.
##'   \item Strobl, C., Boulesteix, A.-L., Zeileis, A., Hothorn, T. (2007) Bias in random forest variable importance measures: Illustrations, sources and a solution. BMC Bioinformatics, 8, 25.
##'   \item Wright, M. N. & Ziegler, A. (2017). "ranger: A fast implementation of random forests for high dimensional data in C++ and R". J Stat Softw 77:1-17, <\doi{10.18637/jss.v077.i01}>.
##'   }
##'
##' @name diversityForest-package
##' @aliases diversityForest
NULL