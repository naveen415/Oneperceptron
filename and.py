from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

def main(data, eta, epochs,filename,plotname):

    df = pd.DataFrame(data)

    print(df)

    X,y = prepare_data(df)

    model_AND = Perceptron(eta, epochs)
    model_AND.fit(X, y)

    _ = model_AND.total_loss()

    save_model(model_AND,filename)

    save_plot(df,plotname,model_AND)

if __name__ == '__main__':
    AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    main(data=AND, eta=ETA, epochs=EPOCHS, filename='and.model', plotname='and.png' )