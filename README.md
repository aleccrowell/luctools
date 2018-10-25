luctools: Analysis of Circadian Luciferase Time Series
======================================================

luctools provides a toolset for plotting, detrending and calculating period and phase of luciferase based circadian expression experiments.

----------
Motivation
----------

To improve the ease of use and reproducibility of analysing circadian luciferase time series.

--------
Features
--------

* Plotting of time series with mean and CIs calculated across replicates.

* Detrending of signal for both decay and baseline.

* Calculation and plotting of period and phase.

-------------
Example Usage
-------------

```python
from luctools import analyse

#reads in data and sampling rate of experiment (samples/hour)
data = analyse.luctraces(path_to_input_data,sample_rate)
#plots undetrended data
data.gen_tsplot(output_path)
```
![ImageRelative](data/test.png "undetrended")

```python
#detrends data
data.detrend()
#plots detrended data
data.gen_tsplot(output_path)
```
![ImageRelative](data/test2.png "detrended")

```python
#calculates periods of each time series
data.get_periods()
#calculates phases of each time series
data.get_phases()
#plots period vs phase for each time series
data.gen_phase_plot(output_path)
```
![ImageRelative](data/test_phase_v_period.png "period_phase")

### A Note on Data Formatting
luctools expects input files to be formatted as comma seperated.  The first column should index the observations.  The header should start with any string and the rest of the header should contain the genotype of the observations 
in the column.  Replicates should have the same column headers. 

| Frame | Control | Control | Control | Experimental | Experimental | Experimental |
|---|---|---|---|---|---|---|
| index# | data | data | data | data | data | data |

------------
Installation
------------

pip install luctools

----
TO DO
----

* Implement statistical comparison of phases between groups.

* Add api reference on readthedocs.

----
Built With
----

* numpy
* pandas
* scipy 
* matplotlib
* seaborn
* peakutils

-------
License
-------

© 2018 Alexander M. Crowell: BSD-3
