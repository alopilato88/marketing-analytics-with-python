##################
# Bass Diffusion #
##################

## Set Seed
set.seed(1)

## Read in the sales data
sales <- read.csv(file.choose())  ## Choose the file sales_data.csv

## Create cumulative sales and lag cumulative sales variables
sales$cumsales <- cumsum(sales$sales)
sales$cumsales_1 <- c(0, sales$cumsales[1:length(sales$cumsales)-1])
sales$cumsales2 <- sales$cumsales^2
sales$cumsales2_1 <- c(0, sales$cumsales2[1:length(sales$cumsales2)-1])

## Run the Bass Diffusion Model regression
bass <- lm(sales ~ cumsales_1 + cumsales2_1, data = sales)
summary(bass)

## Determine m, p, and q
a <- bass$coeff[1]  ## Intercept
b <- bass$coeff[2]  ## Coefficient on cumsales_1
c <- bass$coeff[3]  ## Coefficient on cumsales2_1
N1 <- (-b+sqrt(b^2-4*a*c))/(2*c)
N2 <- (-b-sqrt(b^2-4*a*c))/(2*c)
N <- max(N1,N2); names(N) <- "N"
p <- a/N; names(p) <- "p"
q <- b+p; names(q) <- "q"
print(N)
print(p)
print(q)

## Forecast the sales
periods <- 50
t <- seq(1, periods)
forecasts <- data.frame(t=1:periods)
psales <- double(periods); pcumsales <- double(periods+1)
for (i in 1:periods){
  psales[i] <- p*N+(q-p)*pcumsales[i]-(q/N)*pcumsales[i]^2
  pcumsales[i+1] <- pcumsales[i]+psales[i]
}
forecasts$psales <- psales
forecasts$pcumsales <- pcumsales[-1]  ## Remove the first value which is 0

## Export the two data sets
write.csv(sales, file.choose(new=TRUE), row.names = FALSE) ## Name the file cumulative_sales_data.csv
write.csv(forecasts, file.choose(new=TRUE), row.names = FALSE) ## Name the file sales_forecasts.csv
