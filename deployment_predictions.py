import datarobot as dr
import pandas as pd
import os


def submit_csv_batch(
    deployment_id: str,
    input_file: str,
    output_file: str,
    max_explanations_returned: int,
    max_wait: int,
) -> pd.DataFrame:

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
