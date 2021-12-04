from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from TaxiFareModel.data import get_data, get_Xy, clean_data, holdout
from TaxiFareModel.encoders import DistanceToCenter, DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
import random
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from google.cloud import storage
import TaxiFareModel.params as params

MLFLOW_URI = "https://mlflow.lewagon.co/"

EXPERIMENT_NAME = f"[SG] [SG] [alejandro] linear regression + version {random.randint(1,100)}"  # 🚨 replace with your country code,
#city, github_nickname and model name and version

class Trainer():
    def __init__(self, X, y, model_name):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.model_name = model_name
        self.experiment_name = EXPERIMENT_NAME
        self.rmse = None
        self.metrics_dict = None

    def model_selector(self):
        model_dict = {}
        if self.model_name == 'lasso':
            model_dict['lasso'] = Lasso()
        if self.model_name == 'ridge':
            model_dict['ridge']= Ridge()
        if self.model_name == 'decision_tree':
            model_dict['decision_tree'] = DecisionTreeRegressor()
        else:
            model_dict['linear_regressor'] = LinearRegression()
        return model_dict


    def set_pipeline(self, dtc = True):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        dist_pipe = Pipeline([
                ('dist_trans', DistanceTransformer()),
                ('stdscaler', StandardScaler())
            ])

        # if dtc, create another pipe
        if dtc == 1:
            dist_to_center_pipe = Pipeline([
                ('dist_to_center', DistanceToCenter()),
                                  ('stdscaler', StandardScaler())])

        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        # create preprocessing pipeline

        if dtc == 1:
            preproc_pipe = ColumnTransformer(
                [('distance', dist_pipe, [
                    "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
                    'dropoff_longitude'
                ]),
                 ('distance_to_center', dist_to_center_pipe, [
                     "pickup_latitude", "pickup_longitude"
                 ]), ('time', time_pipe, ['pickup_datetime'])],
                remainder="drop")

        elif dtc == 0:
            preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
                "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
                'dropoff_longitude'
            ]), ('time', time_pipe, ['pickup_datetime'])],
                                             remainder="drop")


        model_dict = self.model_selector()
        model = model_dict[self.model_name]
        # Add the model of your choice to the pipeline
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            (self.model_name, model)
        ])
        self.pipeline = pipe

    def run(self,dtc):
        """set and train the pipeline"""
        self.set_pipeline(dtc)
        # train the pipelined model
        self.pipeline.fit(self.X, self.y)

    def run_CV(self,dtc):
        """set and train the pipeline"""
        self.set_pipeline(dtc)
        # train the pipelined model
        self.pipeline.fit(self.X, self.y)
        cv = cross_validate(self.pipeline, self.X, self.y,
                            scoring=['r2','neg_root_mean_squared_error'])
        self.metrics_dict = {'r2':cv['test_r2'].mean(), 'rmse':(-1.)*cv['test_neg_root_mean_squared_error'].mean()}
        print(self.metrics_dict)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        # compute y_pred on the test set
        y_pred = self.pipeline.predict(X_test)
        # call compute_rmse
        rmse = compute_rmse(y_pred, y_test)
        print(f"Computed rmse of {rmse}")
        self.rmse = rmse

######

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')

    def upload_model_to_gcp(self):
        """ Takes joblib and uploads to gcloud bucket"""
        client = storage.Client()
        bucket = client.bucket(params.GCLOUD_BUCKET_NAME)
        blob = bucket.blob(params.GCLOUD_MODEL_STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')

if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    X,y = get_Xy(df)
    X_train, X_test, y_train, y_test = holdout(X,y)

    dtc_values = [0,1]
    models = ['lasso','ridge','decision_tree']
    for dtc in dtc_values:
        print(f'DTC is {dtc}')
        for model_name in models:
            print(f'Currently running model {model_name}')
            trainer = Trainer(X_train, y_train, model_name)
            # trainer.set_pipeline()
            trainer.run_CV(dtc)
            trainer.evaluate(X_test, y_test)
            print('model', model_name + ' dtc ' + str(dtc))
            trainer.mlflow_log_param('model', model_name + ' dtc '+ str(dtc))
            for key,val in trainer.metrics_dict.items():
                trainer.mlflow_log_metric(key, val)
            trainer.save_model()
            trainer.upload_model_to_gcp()
    print('done')
