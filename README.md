# Airline Tweet Sentiment Analysis
This project builds and deploys a model that predicts sentiment of tweets that are fed into it. This is meant to serve as the final steps of a broader demo that includes:
- Streaming in tweet data via Data Flow
- Securing and governing the tweet data with SDX
- Predicting against the tweet data
- Enabling business users with a visual application to give interactive access to the predictions

A recording of the demo that covers this portion can be found [here](https://drive.google.com/open?id=1SAbFsQk6AWAS4tWX3O8jdkwhM1lz-L7s).

## Initial Setup
There are two main elements that are required from a setup perspective for this demo.

### Training Data
In creating a sentiment analysis model, the training is run against the [Sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140). This enables the model to be trained and validated against a control set of data, while then able to be used for predictions against the airline tweet dataset.

This data is currently expected to exist in the `util/data` directory. This demo may be improved by instead pulling this data from the data lake, or from an existing table in a data warehouse.

### Visual Application
In order to set up the visual application, you will need to follow [these steps](https://docs.google.com/document/d/1NbKGsv0aeZ2T6VaYyoNxMPniX1hE553A-9X9EIeyqa0/edit) inside of this project.

The initial reference data set for the visual application can be found in the `util/data` directory. This can be further improved by instead running predictions against tweet data, saving those predictions, and then pulling this data from the data lake or data warehouse to be represented in the visual application.
