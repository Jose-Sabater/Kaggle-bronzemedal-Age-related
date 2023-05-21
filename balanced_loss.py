def balanced_logarithmic_loss(y_true, y_pred):
    N = len(y_true)

    # Nc is the number of observations
    N_1 = np.sum(y_true == 1, axis=0)
    N_0 = np.sum(y_true == 0, axis=0)

    # wc prevalence
    prev_w_1 = N_1 / N
    prev_w_0 = N_0 / N

    # wc is equal to the inverse prevalence of c
    w_1 = 1 / prev_w_1
    w_0 = 1 / prev_w_0

    # In order to avoid the extremes of the log function, each predicted probability 𝑝 is replaced with max(min(𝑝,1−10−15),10−15)
    y_pred = np.maximum(np.minimum(y_pred, 1 - 1e-15), 1e-15)

    # balanced logarithmic loss
    loss_numerator = -(w_0 / N_0) * np.sum((1 - y_true) * np.log(1 - y_pred)) - (
        w_1 / N_1
    ) * np.sum(y_true * np.log(y_pred))
    loss_denominator = w_0 + w_1

    return loss_numerator / loss_denominator
