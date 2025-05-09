% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plot.multifor.R
\encoding{UTF-8}
\name{plot.multifor}
\alias{plot.multifor}
\title{Plot method for \code{multifor} objects}
\usage{
\method{plot}{multifor}(x, plot_type = c("both", "density", "boxplot")[1], num_best = 5, ...)
}
\arguments{
\item{x}{Object of class \code{multifor}.}

\item{plot_type}{Plot type, one of the following: "both" (the default), "density", "boxplot".  If "density", kernel density estimate-based plots are produced, if "boxplot", boxplot plots are produced, and if "both", both kernel density estimate-based plots and boxplot plots are produced. See the 'Details' section of \code{\link{plotMcl}} for details.}

\item{num_best}{The number of variables with largest class-focused VIM values to plot. Default is 5.}

\item{...}{Further arguments passed to or from other methods.}
}
\value{
A ggplot2 plot.
}
\description{
Plot function for \code{multifor} objects that allows to obtain a first overview of the result of the
class-focused VIM analysis. This function visualises the distribution of the class-focused VIM values
together with that of the corresponding discriminatory VIM values and
the estimated dependency structures of the multi-class outcome on the variables 
with largest class-focused VIM values. These estimated dependency structures are visualised
using kernel density estimate-based plots and/or boxplots.
}
\details{
In the plot showing the distribution of the class-focused VIM values along with 
that of the discriminatory VIM values, the discriminatory VIM values are 
normalized to make them comparable to the class-focused VIM values. This is 
achieved by dividing the discriminatory VIM values by their mean and multiplying 
it by that of the class-focused VIM values.\cr
For details on the plots of the estimated dependency structures of the 
multi-class outcome on the variables, see \code{\link{plotMcl}}.
The latter function allows to visualise these estimated dependency structures
for arbitrary variables in the data.
}
\examples{
\dontrun{

## Load package:

library("diversityForest")



## Set seed to make results reproducible:

set.seed(1234)



## Construct random forest and calculate class-focused and discriminatory VIM values:

data(hars)
model <- multifor(dependent.variable.name = "Activity", data = hars, 
                  num.trees = 100, probability=TRUE)

# NOTE: num.trees = 100 (in the above) would be likely too small for practical 
# purposes. This small number of trees was simply used to keep the
# runtime of the example short.
# The default number of trees is num.trees = 5000 for datasets with a maximum of
# 5000 observations and num.trees = 1000 for datasets larger than that.



## By default the estimated class-specific distributions of the num_best=5
## variables with the largest class-focused VIM values are plotted:

plot(model)

## Consider only the 2 variables with the largest class-focused VIM values:

plot(model, num_best = 2)

## Show only the density plots or only the boxplots:

plot(model, plot_type = "density", num_best = 2)
plot(model, plot_type = "boxplot", num_best = 2)

## Show only the plot of the distributions of the class-focused and
## discriminatory VIM values:

plot(model, num_best = 0)

}

}
\references{
\itemize{
  \item Hornung, R. (2022). Diversity forests: Using split sampling to enable innovative complex split procedures in random forests. SN Computer Science 3(2):1, <\doi{10.1007/s42979-021-00920-1}>.
  }
}
\seealso{
\code{\link{plotMcl}}
}
\author{
Roman Hornung
}
