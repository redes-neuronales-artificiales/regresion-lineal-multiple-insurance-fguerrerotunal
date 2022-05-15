"""
Regresión Lineal Multiple
-----------------------------------------------------------------------------------------

En este laboratorio se entrenara un modelo de regresión lineal multiple que incluye la 
selección de las n variables más relevantes usando una prueba f.

"""
# pylint: disable=invalid-name
# pylint: disable=unsubscriptable-object

import pandas as pd


# Lea el archivo `insurance.csv` y asignelo al DataFrame `df`
df = pd.read_csv("insurance.csv")

# Asigne la columna `charges` a la variable `y`.
y = df["charges"].values

# Asigne una copia del dataframe `df` a la variable `X`.
X = df.copy()

# Remueva la columna `charges` del DataFrame `X`.
X.drop(["charges"],axis=1, inplace=True)

print(X.shape)
print(y.shape)

# Importe train_test_split
from sklearn.model_selection import train_test_split


# Divida los datos de entrenamiento y prueba. La semilla del generador de números
# aleatorios es 12345. Use 300 patrones para la muestra de prueba.
(X_train, X_test, y_train, y_test,) = train_test_split(
    X,
    y,
    test_size=300,
    random_state=12345,
)

print(X_train.sex.value_counts().to_dict())
print(X_test.sex.value_counts().to_dict())
print(X_train.region.value_counts().to_dict())
print(X_test.region.value_counts().to_dict())
print(y_train.sum().round(2))
print(y_test.sum().round(2))


from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

pipeline = Pipeline(
    steps=[
        # Paso 1: Construya un column_transformer que aplica OneHotEncoder a las
        # variables categóricas, y no aplica ninguna transformación al resto de
        # las variables.
        (
            "column_transfomer",
            make_column_transformer(
                (
                    OneHotEncoder(),
                    make_column_selector(dtype_include = object),
                ),
                remainder="passthrough",
            ),
        ),
        # Paso 2: Construya un selector de características que seleccione las K
        # características más importantes. Utilice la función f_regression.
        (
            "selectKBest",
            SelectKBest(score_func=f_regression),
        ),
        # Paso 3: Construya un modelo de regresión lineal.
        (
            "linearRegression",
            LinearRegression(),
        ),
    ],
)
# Cargua de las variables.

# Defina un diccionario de parámetros para el GridSearchCV. Se deben
# considerar valores desde 1 hasta 11 regresores para el modelo
param_grid = {
    "linearRegression__fit_intercept": list(range(1,12)),#____: ____(____, ____),
}

# Defina una instancia de GridSearchCV con el pipeline y el diccionario de
# parámetros. Use cv = 5, y como métrica de evaluación el valor negativo del
# error cuadrático medio.
gridSearchCV = GridSearchCV(
    estimator = pipeline,
    param_grid = param_grid,
    cv = 5,
    scoring = "neg_mean_squared_error",
    refit = True,
    return_train_score = True,
)

# Búsque la mejor combinación de regresores
print(gridSearchCV.fit(X_train, y_train))

print(gridSearchCV.score(X_train, y_train).round(2))
print(gridSearchCV.score(X_test, y_test).round(2))

from sklearn.metrics import mean_squared_error

# Evalúe el modelo con los conjuntos de entrenamiento y prueba.
y_train_pred = gridSearchCV.predict(X_train)
y_test_pred = gridSearchCV.predict(X_test)

# Compute el error cuadratico medio de entrenamiento y prueba. Redondee los
# valores a dos decimales.

mse_train = mean_squared_error(
    y_train,
    y_train_pred,
).round(2)

mse_test = mean_squared_error(
    y_test,
    y_test_pred,
).round(2)


print(mse_train)
print(mse_test)
