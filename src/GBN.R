
library(bnlearn)



mydata <- read.csv("/Users/bill/Documents/PGMProject/src/data/resized_data_scaled_128.csv")
#mydata[] <- lapply( mydata, factor)
mydata$y <- as.factor(mydata$y)

col_names <- colnames(mydata)
nvars <- ncol(mydata)
n_rows <- sqrt((nvars-1)/3)
n_cols <- sqrt((nvars-1)/3)


dag = empty.graph(col_names)

adj = matrix(0L, ncol = nvars, nrow = nvars,
             dimnames = list(col_names, col_names))

# simple naive bayes
for (i in 2:nvars) {
  adj[col_names[1], col_names[i]] = 1L
  row_n = (i-1) / 3 / n_rows
  col_n = (i-1) / 3 %% n_cols
  #if (i + n_cols*3 <= nvars) {
  #  adj[col_names[i], col_names[i + n_cols*3]] = 1L
  #}
  #if (i + n_rows*3 + 3 <= nvars) {
  #  adj[col_names[i], col_names[i + n_rows*3 + 3]] = 1L
  #}
  #if (i+3 <= nvars){
  #  adj[col_names[i], col_names[i+3]] = 1L
  #}
}


amat(dag) = adj

nparams(dag, mydata)

#graphviz.plot(dag)

fit = bn.fit(dag, mydata)

k <- 3
xval <- bn.cv(mydata, bn=dag, loss="pred-lw-cg", loss.args=list(target='y'), method="hold-out", k=k, m=908)
OBS = unlist(lapply(xval, `[[`, "observed"))
PRED = unlist(lapply(xval, `[[`, "predicted"))
sum(OBS==PRED, na.rm = TRUE) / (908*k)
1 - sum(OBS==PRED, na.rm = TRUE) / (908*k)
mean <- 1-(xval[[1]]$loss + xval[[2]]$loss + xval[[3]]$loss)/k
std <- sd(c(xval[[1]]$loss, xval[[2]]$loss, xval[[3]]$loss))
