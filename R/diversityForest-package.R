##' Diversity Forests
##' 
##' The diversity forest algorithm is a split-finding approach that allows complex split procedures 
##' to be realized in random forest variants. This is achieved by drastically reducing the numbers 
##' of candidate splits that need to be evaluated for each split. The algorithm also avoids the 
##' well-known variable selection bias in conventional random forests that has the effect that variables 
##' with many possible splits are selected too frequently for splitting (Strobl et al., 2007). For 
##' details, see Hornung (2022).
##'
##' This package currently features three types of diversity forests:
##' \itemize{
##' \item the \emph{basic form} of diversity forests that uses univariable, binary splitting, which is also used
##' in conventional random forests
##' \item \emph{interaction forests (IFs)} (Hornung & Boulesteix, 2022), which use bivariable splitting to model quantitative and qualitative interaction effects.
##' IFs feature the \emph{Effect Importance Measure (EIM)}, which ranks the variable pairs with respect to the predictive importance
##' of their quantitative and qualitative interaction effects. The individual variables can be ranked as well
##' using EIM. For details, see Hornung & Boulesteix (2022).
##' \item \emph{multi forests (MuFs)} (Hornung & Hapfelmeier, 2024), a diversity forest-variant for multi-class outcomes. MuFs use both multi-way 
##' and binary splitting. The latter form the basis for the \emph{multi-class variable importance measure (VIM)} and the 
##' \emph{discriminatory VIM} associated with MuFs. The multi-class VIM measures the degree to which the variables are 
##' specifically associated with one or several of the outcome classes. In contrast, the discriminatory VIM, 
##' similar to conventional VIMs, measures the general influence of the variables regardless of their 
##' specific association with individual classes.
##' }
##' Diversity forests with univariable, binary splitting can be constructed using the function \code{\link{divfor}}, 
##' interaction forests using the function \code{\link{interactionfor}}, and multi forests using the function
##' \code{\link{multifor}}. Except for multi forests, which are tailored for multi-class outcomes, all included
##' diversity forest variants support categorical, metric, and survival outcomes.
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
##'   \item Hornung, R. (2022). Diversity forests: Using split sampling to enable innovative complex split procedures in random forests. SN Computer Science 3(2):1, <\doi{10.1007/s42979-021-00920-1}>.
##'   \item Hornung, R., Boulesteix, A.-L. (2022). Interaction forests: Identifying and exploiting interpretable quantitative and qualitative interaction effects. Computational Statistics & Data Analysis 171:107460, <\doi{10.1016/j.csda.2022.107460}>.
##'   \item Hornung, R., Hapfelmeier, A. (2024). Multi forests: Variable importance for multi-class outcomes. arXiv:2409.08925, <\doi{10.48550/arXiv.2409.08925}>.
##'   \item Strobl, C., Boulesteix, A.-L., Zeileis, A., Hothorn, T. (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution. BMC Bioinformatics 8:25, <\doi{10.1186/1471-2105-8-25}>.
##'   \item Wright, M. N., Ziegler, A. (2017). ranger: A fast Implementation of Random Forests for High Dimensional Data in C++ and R. Journal of Statistical Software 77:1-17, <\doi{10.18637/jss.v077.i01}>.
##'   }
##'
##' @name diversityForest-package
##' @aliases diversityForest
NULL