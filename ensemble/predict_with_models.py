def predict_with_efficientnet(image_data):
    processed_image = preprocess_image(image_data)
    processed_image = np.expand_dims(processed_image, axis=0)
    return model_efficientnet.predict(processed_image)


def predict_with_inception(image_data):
    processed_image = preprocess_image(image_data)
    processed_image = np.expand_dims(processed_image, axis=0)
    return model_inception.predict(processed_image)


def predict_with_resnet(image_data):
    processed_image = preprocess_image(image_data)
    processed_image = np.expand_dims(processed_image, axis=0)
    return model_resnet.predict(processed_image)


def predict_with_all_models(image_data):
    # 각 모델로부터 예측 결과를 받음
    predictions_efficientnet = predict_with_efficientnet(image_data)
    predictions_inception = predict_with_inception(image_data)
    predictions_resnet = predict_with_resnet(image_data)

    left_prediction = predictions_efficientnet[0][0][1]
    right_predictions = [
        predictions_efficientnet[1][0][1],
        predictions_resnet[1][0][1],
        predictions_inception[1][0][1]
    ]
    right_prediction = np.mean(right_predictions)

    return {
        "left": float(left_prediction),
        "right": float(right_prediction)
    }
