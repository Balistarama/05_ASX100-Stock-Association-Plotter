<div align="center">
  <img src="https://www.asxonline.com/marketinfo/images/top_asx_logo.gif"><br>
</div>

# 05_ASX100 Stock Association Plotter
Using historical Yahoo Finance data, this project uses the example code set out
at http://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html#sphx-glr-auto-examples-applications-plot-stock-market-py
to plot the ASX100 stock associations.

Given the raw finance data from mid 2016 until the end of 2016 (6 months), the 
program discovers which companies are correlated using sparse inverse covariance estimation.
It then groups these correlated companies together on a 2D plot. The thicker the lines between
two companies, the higher the covariance estimation.

##INSTRUCTIONS:
Ensure you have matplotlib, numpy and sklearn libraries installed and then run the main python file:
```
balistarama@Computer:~/05_ASX100 Stock Association Plotter$ python3 plot_ASX100.py
```

##RESULTS:
As can be seen in the image below, all 100 companies are successfully plotted and we
can see some very common sense correlations such as all the major banks being grouped 
together as well as many mining/coal companies also being grouped together.

<div align="center">
  <img src="https://raw.githubusercontent.com/Balistarama/05_ASX100-Stock-Association-Plotter/master/images/ASX100.png"><br>
</div>

##REFERENCES:
The vast bulk of this code was taken from the Scikit Learn website example found here:
http://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html#sphx-glr-auto-examples-applications-plot-stock-market-py
