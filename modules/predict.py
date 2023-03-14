import json
import dill
import pandas as pd
import os
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')


def predict():
    dirname = f'{path}/data/models/'
    files = os.listdir(dirname)
    dirname2 = f'{path}/data/test/'
    files2 = os.listdir(dirname2)
    pred_list = []
    id_list = []

    with open(f'{path}/data/models/{files[0]}', 'rb') as file:
        model = dill.load(file)
    for j in files2:
        with open(f'{path}/data/test/{j}') as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            pred_list.append(y[0])
            id_list.append(form["id"])

            pred_dict = {'car_id': id_list, 'pred': pred_list}
            pd.DataFrame(pred_dict).to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()