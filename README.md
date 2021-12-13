# About the Project

## Project Goals

My goal is to identify key drivers for tax value for single family properties so that we can improve model accuracy.

## Project Description

A home is often the largest and most expensive purchase a person makes in his or her lifetime. Ensuring homeowners have a trusted way to monitor this asset is incredibly important. One considers several aspects while purchasing a home, the size, how many rooms are available, and many more.

Zillow is a popular estimator for house evaluation available online.  Zillow's Zestimate allows the homebuyers to search for a home that satisfies their location, area, budget, etc.

In this project we want to predict the property tax assessed values ('taxvaluedollarcnt') for single family properties. The focus will be the single unit properties that had a transaction during 2017.


### Initial Questions

- What is the relationship between bedroom count and taxvaluedollarcount?
    - Is it a linear relationship or is there no relationship?
    
- What is the relationship between bathroom count and taxvaluedollarcount?
    - Is it a linear relationship or is there no relationship?

- What is the relationship between square feet and taxvaluedollarcount?
    - Is it a linear relationship or is there no relationship?




### Data Dictionary

| Variable            |     Count and Dtype  |
| ----------------    | ------------------ |
|bedrooms             | 77614 non-null  float64 |
|bathrooms            | 77614 non-null  float64 |
|square_feet          | 76502 non-null  float64 |
|taxes                | 77103 non-null  float64 |
|home_value           | 76689 non-null  float64 |
|propertylandusedesc  | 77614 non-null  object  |
|fips_number          | 77614 non-null  float64 |
|zip_code             | 77339 non-null  float64 |



## Steps to Reproduce

- Create an env.py file that contains the hostname, username and password of the mySQL database that contains the zillow table. Store that env file locally in the repository.
- Clone my repo (including an acquire.py and prepare.py) (confirm .gitignore is hiding your env.py file)
- Libraries used are pandas, matplotlib, seaborn, numpy, sklearn.
- Document conclusions, takeaways, and next steps in the Final Report Notebook.

### Plan

Plan - Acquire - Prepare - Explore - Model - Deliver

- Wrangle
    - Acquire data by using a SQL query to Zillow table in the mySQL database.
    - Prepare data by doing a cleanup of null values, duplicates, removed unnecessary outliers.
    - We will create a function that we can reference later to acquire and prepare the data by storing the function in a file name wrangle.py
    - We will split our data to a train, validate, and test
- Explore
    - Create a visualizations correlating to hypotheses statements
    - Run at least two statistical tests that will support whether the hypothesis has been rejected or not
- Modeling