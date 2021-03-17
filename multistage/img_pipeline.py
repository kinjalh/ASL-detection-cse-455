import cv2
import numpy as np
import torch
from typing import Tuple
import src.asl_nn as nn
import src.asl_img_process as preproc

TRAIN_CSV = "data/sign_mnist_train.csv"
TEST_CSV = "data/sign_mnist_test.csv"

def create_training_tensors(
    csv_file,
) -> Tuple[torch.tensor, torch.tensor]:
    x_np, y_np = preproc.load_data_from_csv(csv_file, (28, 28))
    n = np.shape(x_np)[0]

    y_enc = one_hot_encode(y_np, 25)

    procd_imgs = np.zeros((n, 28, 28))
    i = 0
    for img in x_np:
        procd_imgs[i] = preproc.preprocess_img(img)
        i += 1    

    x_flat = np.zeros((n, 784))
    i = 0
    for img in procd_imgs:
        x_flat[i] = np.reshape(procd_imgs[i], (784,))
        i += 1
    
    x_tensor = torch.tensor(x_flat)
    y_tensor = torch.tensor(y_enc)
    return x_tensor.float(), y_tensor.float()


def one_hot_encode(
    y: np.ndarray,
    n_encodings: int
) -> np.ndarray:
    y = y.astype(np.int32)
    n = np.shape(y)[0]
    res = np.zeros((n, n_encodings))
    i = 0
    for val in y:
        res[i][val] = 1
        i += 1
    return res


def compute_acc(
    y_true: torch.tensor,
    y_pred: torch.tensor
) -> float:
    y_t_np = y_true.clone().detach().numpy()
    y_p_np = y_pred.clone().detach().numpy()
    n = np.shape(y_t_np)[0]
    count = 0
    for i in range(0, n):
        if np.argmax(y_p_np[i]) == np.argmax(y_t_np[i]):
            count += 1
    return count / n


def train_model(
    model: nn.MSModel,
    x_data: torch.tensor,
    y_data: torch.tensor,
    epochs: int,
    batch_size: int,
    save_file: str
) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    n = x_data.shape[0]
    for e in range(0, epochs):
        loss_avg = 0
        acc_avg = 0
        for i in range(0, n, batch_size):
            x_train = x_data[i:i + batch_size]
            y_train = y_data[i:i + batch_size]
            m = x_train.shape[0]

            y_pred = model.forward(x_train)
            y_true = torch.argmax(y_train, dim=1)

            train_loss = criterion(y_pred, y_true)
            acc = compute_acc(y_pred, y_train)
            loss_avg += train_loss * m / n
            acc_avg += acc * m / n

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()


        print('loss = ', loss_avg, ' acc = ', acc_avg, ' e = ', e, ' n = ', n)
    torch.save(model, save_file)


def test_model_acc(
    model: nn.MSModel,
    test_csv: str,
    prefix: str = None
) -> None:
    x_test, y_test = create_training_tensors(test_csv)
    y_pred = model.forward(x_test)
    acc = compute_acc(y_test, y_pred)
    if prefix:
        print(prefix, ' n = ', x_test.shape[0], ' acc = ', acc)
    else:
        print('n = ', x_test.shape[0], ' acc = ', acc)


if __name__ == "__main__":
    # model = nn.MSModel()
    # x_tensor, y_tensor = create_training_tensors(TRAIN_CSV)

    # train_model(
    #     model=model,
    #     x_data=x_tensor,
    #     y_data=y_tensor,
    #     epochs=200,
    #     batch_size=128,
    #     save_file="model.pt"
    # )

    model = torch.load("model_multistage.pt")
    test_model_acc(model, TRAIN_CSV, 'TRAIN')
    test_model_acc(model, TEST_CSV, 'TEST')
