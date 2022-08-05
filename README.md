# Prediction Explanation Helper Pipeline

### Introduction

This repo contains code designed to help users interact prediction explanations in the datarobot API. The overall design is that the main class stores the data in a pipleine and performs actions on this data inplace. As such everything reolves around a class titled <code>DR_Pred_Explan_Pipeline</code> in **dr_api_request.py** which contains a rather large number of class methods which perform all of the needed processing

The code is designed to function in a foward pipe operator sort of style meaning you can write the following:

<code>
pred_explan_pipe = DR_Pred_Explan_Pipeline(connection=dr_connection)

pred_explan_pipe.load_data_from_csv("./data/sampledata.csv").deployment_request(deployment_id="").process_deployment_explanations_melted() 
</code>

To obtain your result. While somewhat elegant this approach does have some drawbacks - it acts on the data (stored in the data attribute of the class) in place meaning if you make a mistake with a step you will need to go back and redo all steps from the beggining. The package assumes that if the user needs to experminent they will copy the data somewhere to facilitate easier iteration on their code.

The other main drawback is that the organization of the DR_Pred_Explan_Pipeline class contained in the **dr_api_request.py** file is somewhat of a mess the file imports lots of packages and the class has a ton of methods and attributes. This also will not work on data that is too large to store in memory on the system

This code is designed to facilitate easy integration of internal tools through all phases of development as switching from testing the prediction explanations of various model canidates to processing the explanations from a deployed model is as simple as switching <code>project_request</code> with <code>deployment_request</code>


### Package Organization

The main working code is the **dr_api_requests file** - this code contains the higher level pipeline class wrapper which in turn draws apon functions contains in other folders for the sake of easier organization.

**data_sources.py** contains all code to pull from various data sources
**deployment_predictions.py** contains all code to pull predictions along with their explanations from deployments hosted in the datarobot MLOps platform
**project_predictions.py** contains all code to pull predictions along with their explanations from models contained in datarobot projects
**process_explanations.py** contains all the code to process the prediction explanations into more human (and BI tool) digestable forms
###
