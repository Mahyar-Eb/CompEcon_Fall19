library(Multifit)
library(rmgarch)
library(rugarch)
library(readxl)
library(ggplot2)
library(tidyre)
library(dplyr)
library(rSHAPE)
#set directory
setwd("C:/Users/mahyar/Desktop/dataset")
#time series graph for variables
dfGraph <- russia %>%
         select(Date, ruindex) %>%
         gather(key = "variable", value = "value", -Date)
         head(dfGraph, 189)

ggplot(dfGraph, aes(x = Date, y = value)) +
       geom_area(aes(color = variable, fill = variable),
                 alpha = 0.5, position = position_dodge(0.8)) +
       scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
       scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
       labs(title = "monthly russia index from 1996 to 2011")

dfGraph <- russia %>%
         select(Date, oilp) %>%
         gather(key = "variable", value = "value", -Date)
         head(dfGraph, 189)

 ggplot(dfGraph, aes(x = Date, y = value)) +
       geom_area(aes(color = variable, fill = variable),
                 alpha = 0.5, position = position_dodge(0.8)) +
       scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
       scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
       labs(title = "brent oil price from 1996 to 2011")
#estimation of DCC model

Dat = russia[, 2:3, drop = FALSE]
 xspec = ugarchspec(mean.model = list(armaOrder = c(1, 1)), variance.model = list(garchOrder = c(1,1), model = 'eGARCH'), distribution.model = 'norm')
 uspec = multispec(replicate(2, xspec))
 spec1 = dccspec(uspec = uspec, dccOrder = c(1, 1), distribution = 'mvnorm')
 spec1a = dccspec(uspec = uspec, dccOrder = c(1, 1), model='aDCC', distribution = 'mvnorm')
 spec2 = dccspec(uspec = uspec, dccOrder = c(1, 1), distribution = 'mvlaplace')
 spec2a = dccspec(uspec = uspec, dccOrder = c(1, 1), model='aDCC', distribution = 'mvlaplace')
 cl = makePSOCKcluster(2)
 multf = multifit(uspec, Dat, cluster = cl)
fit1 = dccfit(spec1, data = Dat, fit.control = list(eval.se = TRUE), fit = multf, cluster = cl)
fit1a = dccfit(spec1a, data = Dat, fit.control = list(eval.se = TRUE), fit = multf, cluster = cl)
fit2 = dccfit(spec2, data = Dat, fit.control = list(eval.se = TRUE), fit = multf, cluster = cl)
fit2a = dccfit(spec2a, data = Dat, fit.control = list(eval.se = TRUE), fit = multf, cluster = cl)
# First Estimate a QML first stage model (multf already estimated). Then
# estimate the second stage shape parameter.
spec3 = dccspec(uspec = uspec, dccOrder = c(1, 1), distribution = 'mvt')
fit3 = dccfit(spec3, data = Dat, fit.control = list(eval.se = FALSE), fit = multf)
# obtain the multivariate shape parameter:
mvt.shape = rshape(fit3)
# Plug that into a fixed first stage model and iterate :
mvt.l = rep(0, 6)
mvt.s = rep(0, 6)
mvt.l[1] = likelihood(fit3)
mvt.s[1] = mvt.shape
for (i in 1:10) {
     xspec = ugarchspec(mean.model = list(armaOrder = c(1, 1)), variance.model = list(garchOrder = c(1,1), model = 'eGARCH'), distribution.model = 'std', fixed.pars = list(shape = mvt.shape))
     spec3 = dccspec(uspec = multispec(replicate(2, xspec)), dccOrder = c(1,1), distribution = 'mvt')
     fit3 = dccfit(spec3, data = Dat, solver = 'solnp', fit.control = list(eval.se = FALSE))
     mvt.shape = rshape(fit3)
     mvt.l[i + 1] = likelihood(fit3)
     mvt.s[i + 1] = mvt.shape }

# Finally, once more, fixing the second stage shape parameter, and
# evaluating the standard errors
xspec = ugarchspec(mean.model = list(armaOrder = c(1, 1)), variance.model = list(garchOrder = c(1,1), model = 'eGARCH'), distribution.model = 'std', fixed.pars = list(shape = mvt.shape))
spec3 = dccspec(uspec = multispec(replicate(2, xspec)), dccOrder = c(1, 1), distribution = 'mvt', fixed.pars = list(shape = mvt.shape))
fit3 = dccfit(spec3, data = Dat, solver = 'solnp', fit.control = list(eval.se = TRUE), cluster = cl)

#exercise is repeated for the asymmetric DCC (MVT) model.
 xspec = ugarchspec(mean.model = list(armaOrder = c(1,1)), variance.model = list(garchOrder = c(1,1), model = "eGARCH"),  distribution.model = "norm")
 spec3a  = dccspec(uspec = multispec( replicate(2, xspec) ), dccOrder = c(1,1), distribution = "mvt", model="aDCC")
fit3a = dccfit(spec3a, data = Dat, fit.control = list(eval.se=FALSE), fit = multf)
# obtain the multivariate shape parameter:
mvtx.shape = rshape(fit3a)
# Plug that into a fixed first stage model and iterate :
mvtx.l = rep(0, 6)
 mvtx.s = rep(0, 6)
 mvtx.l[1] = likelihood(fit3)
 mvtx.s[1] = mvtx.shape
 for(i in 1:5){
     xspec = ugarchspec(mean.model = list(armaOrder = c(1,1)), variance.model = list(garchOrder = c(1,1), model = "eGARCH"),  distribution.model = "std", fixed.pars = list(shape=mvtx.shape))
     spec3a = dccspec(uspec = multispec( replicate(2, xspec) ), dccOrder = c(1,1), model="aDCC", distribution = "mvt")
    fit3a = dccfit(spec3a, data = Dat, solver = "solnp", fit.control = list(eval.se=FALSE))
     mvtx.shape = rshape(fit3a)
     mvtx.l[i + 1] = likelihood(fit3a)
     mvtx.s[i + 1] = mvtx.shape
 }
# Finally, once more, fixing the second stage shaoe parameter, and evaluating the standard errors
xspec = ugarchspec(mean.model = list(armaOrder = c(1,1)), variance.model = list(garchOrder = c(1,1), model = "eGARCH"),  distribution.model = "std", fixed.pars = list(shape=mvtx.shape))
spec3a = dccspec(uspec = multispec( replicate(2, xspec) ), dccOrder = c(1,1), model="aDCC", distribution = "mvt", fixed.pars=list(shape=mvtx.shape))
fit3a = dccfit(spec3a, data = Dat, solver = "solnp", fit.control = list(eval.se=TRUE), cluster = cl)

#for getting plot

 par(mfrow = c(2,2))
 RR = timeSeries(cbind(R1[2,1,],R2[2,1,],R3[2,1,]), D)
    plot(RR[,1], ylab = "cor", col = colx[1], lty=1 ,lwd=1)
 for(i in 2:3) lines(RR[,i], col = colx[i], lty = i, lwd=1+i/10)
 title("russia stock market index and oil price correlation from1996 to 2011")
