# Starbucks_capstone

### Description
This is the deliverable for the Udacity capstone project.
The detailed description of the analysis and findings can be found on  [this section](https://marco-bellini.github.io/starbucks/) of my personal portfolio website.

### Project overview and motivation
The data set is provided by Starbucks: it contains simulated customer behavior on Starbucks rewards mobile app. 
Each simulated customer receives every few days an offer (with different offer types, durations and rewards) through a variety of channels (social media, email, etc).
Not all customers receive the same offers. Customers may receive multiple offers and offers whose validity periods overlap partially. Finally, not all customers visualize an offer they receive.
The problem is simplified as in this simulated database there is only one product.

### Dependencies and Installation
- SQLalchemy
- pandas, sklearn, seaborn

Most important, please make sure to meet the following dependencies: 

pandas:  0.24.2
sklearn:  0.21.2
statsmodels:  0.9.0


###  Files in the repository

The data engineering and some plotting functions are complex and detreact from the readability of the notebooks.

Libraries for data engineering and plotting:
- capstone_data_eng.py (library used to perform the key data engineering tasks)
- auxiliary.py (auxiliary functions used by capstone_data_eng.py)
- plotting_functions.py (library used to perform complex visualization like 2d bins histograms or plotting the transactions and offers of a customers).

Notebooks (to be viewed in order):
- 01_FE_EDA_offers.ipynb (notebook used for the data exploration of offers)
- 02_FE_EDA_customers.ipynb (notebook used for the data exploration of customers)
- 03_FE_EDA_transactions.ipynb (notebook used for the data exploration of transactions)
- 04_business_analysis.ipynb (notebook used for the statistics-based business analysis)
- 05_classification.ipynb (notebook used for the classification of offers rewarded but not viewed)

###  Files not provided
- json files from Udacitiy

### Results of the analysis

This capsone project was made very interesting and challenging by the hidden complexity. The main challenge is the overlapping of offers. Without a clean separation of data it would be impossible to draw significant conclusions.
Also, to understand the complexity of the situation, custom visualizations are needed.

#### Recommendations
The recommendation for Starbucks would be:
* continue to use offers to increase revenues
* design offers that are non overlapping to improve the predictiviness of the analysis
* consider if there's need to obtain correct age and income information from some customers
* increase the view rate of some offers by combining the social and the web channels

### Acknowledgement
I would like to thank Udacity for a very high level course and Starbucks for an interesting and challenging capstone project.




