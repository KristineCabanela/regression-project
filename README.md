# About the Project

## Project Goals

My goal is to identify key drivers for tax value for single family properties

## Project Description



### Initial Questions

- Do single family properties that have less than 1
- 
- 
- 

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
    - Prepare data by 
    - We will create a function that we can reference later to acquire and prepare the data by storing the function in a file name wrangle.py
- Explore
- Modeling