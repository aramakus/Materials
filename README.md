<h1> Electricity demand in Victoria during coronavirus pandemic </h1>

This repository containes model, datasets and scripts and materials used in the article/report. The code was tested on Ubuntu 18.04 with the following packages:

* Python 3.8.2
* Pytorch 1.6.0 
* Numpy 1.19.1
* Pandas  1.1.1
* Scikit-Learn 0.23.1

<h2>Dataset</h2>
The dataset is compiled from different sources, all of which are public available. The energy data can be found at Australian Energy Market Operator website, the weather data is provided by Australian Bureau of Meteorology. List of public holiday, including state holidays, was included. School terms dates, available at Education Victoria website, were adjusted for 2020 to account for disruptions due to various phases of lockdown. The most precise source of this seems to be tweeter feed of Daniel Andrews...
Script, used for scraping the data and combining it in a ready-to-use dataset can be found in "data scraping" folder.

<h2>Feature Engineering</h2>
The data from AEMO comes in 30 minute intervals, so 48 points per day. Data exploration shows that there are day, during which intraday prices go negative. The following features were added to the dataset:

* Fraction of total daily demand at negative price
* Total positively priced daily demand
* Total negatively priced daily demand (can be 0)
* Positive demand-weighted average daily price
* Negative demand-weighted average daily price (can be 0)

<h2>Assumptions</h2>
Construction of any model requies assumptions. Here is a list of the main assumptions used in this model.

* Lowest and highest daily temperatures are known with a perfect accuracy for 7 days ahead.
* Weather data from only a single weather station (Olympic Park near Melbourne CDB) was used. It provides an estimate for the Melbourne region, where 5 out of 6.7 million Victorian residents live and work.
* Announcements, relevant to holidays and remote schooling are known for 7 days prior to the commencement date.

<h3> Model </h3>
The model uses an encoder-decoder architecture. An excellent explanation of use cases and can be found in this [article](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346).  The diagram shows how the model works:

![Image](https://github.com/aramakus/Materials/blob/master/Plots%2B/schema.png) 

The model uses 2 weeks of prior data to produce a forecast for electricity demand for the following 7 days. It was trained on the data from January 2015 to January 2019. Throughout training the data for 2019 was used as a validation set. The model states were saved once a new best value for L1 loss on validation set was achieved (callback). After training, the model was used to produce a forecast for the whole 2019 and for a period from January to October of 2020.  
