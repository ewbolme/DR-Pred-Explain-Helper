import datarobot as dr
import pandas as pd
import os

#TODO batch gives a different response than realtime - check realtime
def submit_csv_batch(
    deployment_id: str,
    input_file: str,
    output_file: str,
    max_explanations_returned: int,
    max_wait: int,
) -> pd.DataFrame:

    """
    A wrapper for a datarobot batch prediction job including prediction explanations

    args:
        deployment_id (str): The ID of the deployment in question to pull the prediction explanations from
        input_file: (str): The csv file to submit to the datarobot deployment (in batches) for scoring - note this is required so the pipeline will temporarily create one
        output_file: (str): A name for the output of the model scoring job - note this is required so the pipeline will temporarily create one
        max_explanations_returned

    returns:
        A pd.Dataframe containing the variables - their scores and the associated explanations for each prediction    

    steps:
        * create a datarobot batch prediction job for the input file
        * wait for the job to finish
        * delete the input file
        * load the output file into memory
        * delete the output file
        * return the file with the scores and predictions
    """

    # TODO add using server helper in here to access alternative DataRobot host.
    job = dr.BatchPredictionJob.score(
        deployment=deployment_id,
        intake_settings={
            "type": "localFile",
            "file": input_file,
        },
        output_settings={
            "type": "localFile",
            "path": output_file,
        },
        # If explanations are required, uncomment the line below
        max_explanations=max_explanations_returned,
        download_timeout=max_wait,
        passthrough_columns_set="all",
        # Uncomment this for Prediction Warnings, if enabled for your deployment.
        # prediction_warning_enabled=True
    )

    job.wait_for_completion()

    os.remove(input_file)
    output_data = pd.read_csv(output_file)
    os.remove(output_file)

    return output_data
