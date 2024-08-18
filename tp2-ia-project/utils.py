from sklearn.datasets import fetch_california_housing

def filter_outliers(df_to_filter):
    fun_df = df_to_filter.copy()
    fun_df = fun_df[(0 <= fun_df["MedInc"]) & (fun_df["MedInc"] <= 14)]
    fun_df = fun_df[(0 <= fun_df["AveRooms"]) & (fun_df["AveRooms"] <= 10)]
    # fun_df = fun_df[(0.7 <= fun_df["AveBedrms"]) & (fun_df["AveBedrms"] <= 1.4)]
    fun_df = fun_df[(0 <= fun_df["Population"]) & (fun_df["Population"] <= 5000)]
    fun_df = fun_df[(0 <= fun_df["AveOccup"]) & (fun_df["AveOccup"] <= 6)]
    fun_df = fun_df[(0 <= fun_df["MedHouseVal"]) & (fun_df["MedHouseVal"] <= 4.99999999)]
    return fun_df


def get_clean_dataset():
    df_california = fetch_california_housing(as_frame=True).frame

    return filter_outliers(df_california.drop(columns=['Longitude', 'AveBedrms']))


class CaliforniaCleanDataset:
    def __init__(self):
        df = get_clean_dataset()
        self.data = df.drop(columns=['MedHouseVal'])
        self.target = df["MedHouseVal"]

