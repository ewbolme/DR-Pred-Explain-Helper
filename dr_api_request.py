from dataclasses import dataclass, field
from pickle import EMPTY_DICT
import pandas as pd
import datarobot as dr
from typing import Optional
from deployment_predictions import submit_csv_batch
from project_predictions import submit_request_to_model
from data_sources import get_from_csv
from enum import Enum
from process_explanations import (
    return_explanations_flat,
    return_melted_dataframe,
)


class Pipeline_Stage(Enum):
    EMPTY = 1
    LOADED_DATA = 2
    PREDICTIONS_OBTAINED = 3
    PREDICTIONS_PROCESSED = 4


@dataclass
class DR_Connection:
    """
    This is a class which handles some of the finer points of creating a client connection to DataRobot

    Attributes:
        api_endpoint (str): the endpoint for the prediction requests - for our cloud server this should be 'https://app.datarobot.com/api/v2'
        api_key (str): an API key obtained from the developer tools section of the DataRobot GUI 
        ssl_insecure (bool): 
        client (datarobot.client): A member of the datarobot.Client class not instantiated on creation of a class member - created through the create_connection method

    Methods:
        create_connection - opens a connection to the DataRobot server addressed by the api_endpoint attribute
        test_connection - tests the DataRobot connection contained in the client attribute
        close_connection - closes the DataRobot connection contained in the client attribute
    """

    api_endpoint: str
    api_key: str
    ssl_insecure: bool
    client: dr.Client = field(init=False)

    def create_connection(self) -> None:
        self.client = dr.Client(
            endpoint=self.api_endpoint,
            token=self.api_key,
            ssl_verify=not self.ssl_insecure,
            user_agent_suffix="IntegrationSnippet-ApiClient",
        )

    def test_connection(self) -> None:
        # TODO code to create connection here
        pass

    def close_connection(self) -> None:
        # TODO make sure there is an actual open DR client in the client attribute:
        self.client.close()


@dataclass
class DR_Pred_Explan_Pipeline:
    """
    This is a class desgined to facilitate the easy processing of prediction explanations from data robot.

    Attributes:
        connection (Dr_Connection): This attribute inherts from the DR_Connection class, 
            it is optional to fill it but reccommended as you may need to close the conneciton and instantiate another one at some point
        data (pd.DataFrame): A pandas dataframe to hold the data for modification in place
        temp_input (str): A string to hold the temporary file path for submission to datarobot for batch scoring - 
            defaults to "temp_file_input.csv"
        temp_output (str): A string to hold the temporary file path the output to datarobot of batch scoring - 
            defaults to "temp_file_output.csv"
        max_explanations (int): A integer to hold the max number of explanations to output defaults to 10 
            (note projects will not output more than 10 explanations)
        shap_bool (bool): A boolean value that holds whether or not to return SHAPly value explanations - 
            defaults to False meaning XEMP explanations will be returned
        last_task_run (str): designed to hold values of the enumerator Pipeline_Stage
        association_id (str): designed to hold the name of an ID column - defaults to None meaning the dataset lacks an ID column

    Methods:
        load_data_from_csv - loads data from a csv into the pipeline storing it in the data attribute
        load_data_from_inmemory - takes a pandas datafame as input storing it in the data attribute
        project_request - makes a request to a model in a project 
        process_deployment_explanations_flat_file - creates a file with a unique column for each important feature - so feature X gets its own column where the column member is the feature's influence on the prediction for the observation in question
        process_deployment_explanations_melted - creates a file with one row per observation / feature joint index - suitable for use in business intellegence tools
        output_explanations_as_json - creates a json output containing the prediction explanations with the following structure
    """

    connection: Optional[DR_Connection] = field(repr=False)
    data: Optional[pd.DataFrame] = field(repr=False)
    temp_input: str = "temp_file_input.csv"
    temp_output: str = "temp_file_output.csv"
    max_explanations: int = 10
    shap_bool: bool = False
    last_task_run: str = Pipeline_Stage.EMPTY
    association_id: str = None

    def load_data_from_csv(self, input_filename: str) -> None:
        """
        Loads data from a csv file into the project pipeline overwriting the self.data attribute

        args:
            input_filename (str): The csv filename ot load into the data attribute as a pandas dataframe
        """
        self.data = get_from_csv(input_filename)
        self.last_task_run = Pipeline_Stage.LOADED_DATA

    def load_data_from_inmemory(self, input_data: pd.DataFrame) -> None:
        """
        Loads data from a csv file into the project pipeline

        args:
            input_data (pandas.DataFame) An in memory pandas dataframe to load the data from
        """
        self.data = input_data
        self.last_task_run = Pipeline_Stage.LOADED_DATA       

    def deployment_request(self, deployment_id: str, max_wait=300) -> None:
        """
        Makes a batch prediction request to a DataRobot deployment - returns a dataframe with predictions and max_explanation prediction explanations
        
        args:
            deployment_id (str): A string representing the deployment ID to pull the prediction explanations from. Defauts to 300
            max_wait (int): The number of seconds to wait for the deployment to produce the prediction explanations - this can be exceeded if the queue for your prediciton server is busy
        """

        # TODO make this check a function so the code doesn't have to be repeated
        if self.last_task_run != Pipeline_Stage.LOADED_DATA:
            print('to function correctly the deployment request method requires a dataset loaded through the load data function - if you have loaded data by overwriting the data attribute manually please also overwrite the last_task_run attribute with the enum Pipeline_Stage.LOADED_DATA')
            return

        self.data.to_csv(self.temp_input, sep=",")
        self.data = submit_csv_batch(
            deployment_id,
            self.temp_input,
            self.temp_output,
            self.max_explanations,
            max_wait,
        )
        self.last_task_run = Pipeline_Stage.PREDICTIONS_OBTAINED

    def project_request(self, project_id: str, model_id: str) -> None:
        """
        Makes a prediction request (along with prediction explanations) to a DataRobot model in a project returns a new pandas dataframe with the predictions and prediction explanations
        
        args:
            project_id (str): the project ID containing the model you desire the prediction explantions from
            model_id (str): the model ID to pull the prediction explanations from 
        """
        
        if self.last_task_run != Pipeline_Stage.LOADED_DATA:
            print('to function correctly the project request method requires a dataset loaded through the load data function - if you have loaded data by overwriting the data attribute manually please also overwrite the last_task_run attribute with the enum Pipeline_Stage.LOADED_DATA')
            return
        
        self.data = submit_request_to_model(
            data=self.data,
            project_id=project_id,
            model_id=model_id,
            max_explanations=self.max_explanations,
            shap_bool=self.shap_bool,
        )
        self.last_task_run = Pipeline_Stage.PREDICTIONS_OBTAINED

    def process_deployment_explanations_flat_file(self) -> None:
        """
        Creates a file with a unique column for each important feature - so feature X gets its own column where the column member is the feature's influence on the prediction for the observation in question
        """

        if self.last_task_run != Pipeline_Stage.PREDICTIONS_OBTAINED:
            print()
            return

        self.data = return_explanations_flat(self.data)
        self.last_task_run = Pipeline_Stage.PREDICTIONS_PROCESSED

    def process_deployment_explanations_melted(self) -> None:
        """
        Creates a file with one row per observation / feature joint index - suitable for use in business intellegence tools
        """

        if self.last_task_run != Pipeline_Stage.PREDICTIONS_OBTAINED:
            print()
            return

        self.data = return_melted_dataframe(self.data)
        self.last_task_run = Pipeline_Stage.PREDICTIONS_PROCESSED

    def output_explanations_as_json(self):
        """
        Makes a prediction request (along with prediction explanations) to a DataRobot model in a project
        """
        print('Not yet implemented')
