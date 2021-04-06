import streamlit as st
import altair as alt
import pydeck as pdk

train_area = st.empty()

"""
# California Housing Prices
This is the California Housing Prices dataset which contains data drawn from the 1990 U.S. Census. The following table provides descriptions, data ranges, and data types for each feature in the data set.

## Let's first take a look at imports
"""

with st.echo():
    import tensorflow as tf
    import numpy as np
    import pandas as pd

"""
## Loading the Dataset
We will use the scikit-learn's dataset module to lead data which is already cleaned for us and only has the numerical feautures. 
"""

with st.echo():
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()

"""
This will load the entire data in the `housing` variable as you can see below
"""

st.subheader('Input Features')
housing.data
st.subheader('Output Lables')
housing.target

"""
## Splitting the data into Train, Test and Dev sets

This is one of the most important thing in beginning of any Machine Learning solution as the result of any model can highly depend on how well you have distributed the data into these sets. 
Fourtunately for us, we have scikit-learn to the rescue where it has become as easy as 2 lines of code.
"""

with st.echo():
    from sklearn.model_selection import train_test_split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full
    )

"""
The `train_test_split()` function splits the data into 2 sets where the test set is 25% of the total dataset. We have used the same function again on the train_full to split it into train and validation set. 25% is a default parameter and you can tweak is as per your needs. Take a look at it from the [Scikit-Learn's Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

## Taking a look at the train data
The colums represent the following data:
"""

st.write(housing.feature_names)

"""
Now let's look at the location of the houses by plotting it on the map using Latitude and Longitude values:
"""

with st.echo():
    map_data = pd.DataFrame(
        X_train,
        columns=[
            'MedInc', 
            'HouseAge', 
            'AveRooms', 
            'AveBedrms', 
            'Population', 
            'AveOccup', 
            'latitude', 
            'longitude'
            ])

    midpoint = (np.average(map_data["latitude"]), np.average(map_data["longitude"]))
    st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 6,
        "pitch": 75,
    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            data=map_data,
            get_position=["longitude", "latitude"],
            radius=1000,
            elevation_scale=4,
            elevation_range=[0, 10000],
            pickable=True,
            extruded=True,
        ),
    ],
))

"""
**Feel free to zoom in or drag while pressing ALT key to change the 3D viewing angle of the map, as required.**

## Preprocessing

As pointed out earlier, this dataset is already well preprocessed by scikit-learn for us to use directly without worrying about any NaN values and other stuff.
Although, we are going to scale the values in specific range by using `StandardScaler` to help our model work effeciently.
"""

with st.echo():
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

"""
## Creating a model
We will be creating a simple Sequential Model with first layer containing 30 neurons and the activation function of RELU.
The next layer will be single neuron layer with no activation function as we want the model to predict a range of values and not just binary or multiclass results like classification problems.
"""
st.sidebar.title('Hyperparameters')
n_neurons = st.sidebar.slider('Neurons', 1, 128, 30)
l_rate = st.sidebar.selectbox('Learning Rate', (0.0001, 0.001, 0.01), 1)
n_epochs = st.sidebar.number_input('Number of Epochs', 1, 50, 20)


with st.echo():
    import tensorflow as tf
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(n_neurons, activation='relu', input_shape=X_train.shape[1:]),
        tf.keras.layers.Dense(1)
    ])

"""
## Compiling the model
Tensorflow keras API provides us with the `model.compile()` function to assign the optimizers, loss function and a few other details for the model.
"""

with st.echo():
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.SGD(l_rate)
    )

"""
## Training the model
In order to train the model you simply have to call the `fit()` function on the model with training and validation set and number of epochs you want the model to train for.

**Try playing with the hyperparameters from the sidebar on the left side and click on the `Train Model` button given below to start the training.**
"""


train = st.button('Train Model')

if train:
    with st.spinner('Training Model...'):
        with st.echo():
            model.summary(print_fn=lambda x: st.write("{}".format(x)))
            
            history = model.fit(
                X_train,
                y_train,
                epochs=n_epochs,
                validation_data=(X_valid, y_valid)
            )
    st.success('Model Training Complete!')

    """
    ## Model Performance
    """

    with st.echo():
        st.line_chart(pd.DataFrame(history.history))

    """
    ## Evalutating the model on Test set

    Again another imortant but easy step to do is to evaluate your model on the test data which it has never seen before. Remember that you should only do this after you are sure enough about the model you'vr built and you should resist making any hyperparameter tuning after evaluating the model on the test set as it would just make it better for test set and again there will be a generalization problem when the model will see new data in production phase.
    """

    with st.echo():
        evaluation = model.evaluate(X_test, y_test)
        evaluation

    """
    > This loss on the test set is a little worse than that on the vakidation set, which is as expected, as the model has never seen the images from test set.
    """

    """
    ## Predictions using the Model
    """

    with st.echo():
        X_new = X_test[:3]
        predictions = model.predict(X_new)

    """
    ### Predictions
    """
    predictions
    """
    ### Ground Truth
    """
    y_test[:3]