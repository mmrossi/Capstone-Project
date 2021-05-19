# Capstone-Project
Final project -- Webpage clustering and categorizing based on page content.

## Preparing environment
If working in conda, open a terminal and navigate to the project folder. Run the following command: conda create --name <your-env-name> --file requirements.txt. This will create an environment with all the modules necessary to run this script. 
Then activate your environment by running: conda activate <your-env-name>.

## Microsoft Azure
+ API cloud deployment in progress.

## FastAPI
+ Prepare the environment (see above).
+ Open the main.py file and run the live server from your terminal: uvicorn main:app --reload
+ Open your browser at http://127.0.0.1:8000/docs#/
+ Input the wikipedia titles that you'd like to cluster in the /urls/{urls} parameter box.
	+ Example: Random forests, Linear Regression, Bitcoin
+ Returns titles and labelled categories
+ To return the cluster image, execute the /image parameter.



Example cluster:

![img](https://github.com/mmrossi/Capstone-Project/blob/main/Cluster%20example.png?raw=true)