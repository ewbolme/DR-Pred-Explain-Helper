import datarobot as dr
import pandas as pd

# TODO pullout code for shap and regular request calls into seperate functions


def submit_request_to_model(
    data: pd.DataFrame,
    project_id: str,
    model_id: str,
    max_explanations: int,
    shap_bool: bool,
) -> pd.DataFrame:
    """A function which obtains prediction explanations for a dataset from a model in a project

    Args:
        data: a pandas dataset with data in the same format the model was trained on
        project_id: the id of the project for the model of intrest
        model_id: the id of the model of intrest
        max_explanations: the number of prediction explanations to return
        shap_bool: a boolean value indicating whether to return shap based prediction explanations

    Returns:
        data: A pandas dataframe containing the prediciton explanations


    * Uploads a dataset to the AI catalog
    * Load the dataset into the project as a project dataset
    * Queues up feature impact and a prediction job (needed to get the explanations)"""

    project = dr.Project.get(project_id=project_id)
    model = dr.Model.get(project=project_id, model_id=model_id)

    if shap_bool:
        model_capabilities = model.get_supported_capabilities()
        # TODO join this back to regular data
        if model_capabilities.get("supportsShap"):
            dataset = dr.Dataset.create_from_in_memory_data(data)
            dataset = project.upload_dataset_from_catalog(dataset_id=dataset.id)
            shap_matrix_job = dr.models.ShapMatrix.create(
                project_id=project.id, model_id=model.id, dataset_id=dataset.id
            )
            shap_matrix = shap_matrix_job.get_result_when_complete()
            return shap_matrix.get_as_dataframe()
        else:
            print("This model does not support shaply values")
            return data

    else:
        dataset = dr.Dataset.create_from_in_memory_data(data)
        dataset = project.upload_dataset_from_catalog(dataset_id=dataset.id)
        try:
            fi_job = model.request_feature_impact()
            fi_job.wait_for_completion()
        except dr.errors.ClientError as err:
            pass
        pred_job = model.request_predictions(dataset.id)
        pred_job.wait_for_completion()

        pei_job = dr.PredictionExplanationsInitialization.create(project_id, model_id)
        pei_job.wait_for_completion()

        if max_explanations > 10:
            print(
                "Project models can only output 10 predictions max setting max explanations to 10"
            )
            max_explanations = 10

        pe_job = dr.PredictionExplanations.create(
            project_id, model_id, dataset.id, max_explanations=max_explanations
        )
        pe = pe_job.get_result_when_complete()
        return pe.get_all_as_dataframe()
