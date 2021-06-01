
import numpy as np
from itertools import product
from custom_loss import CustomLoss1, CustomLoss2, CustomLoss3
from data import read_data
from model import DA_rnn

from tensorflow import summary


def main(debug=False):

    dataroot = 'solar_data.txt'
    
    parameters = dict(  batchsize = [64],
                        nhidden_encoder = [64],
                        nhidden_decoder = [64],
                        ntimestep = [20],
                        lr = [0.001],
                        epochs = [110],
                        loss_func = [CustomLoss1, CustomLoss2, CustomLoss3],
                        diff = [True]
                    )

    param_values = [v for v in parameters.values()]
    print("Number of runs: ", len(list(product(*param_values))))

    run_stats = []
    runs = 0

    for batchsize, nhidden_encoder, nhidden_decoder, ntimestep, lr, epochs, loss_func, diff in product(*param_values):

        # Read dataset
        print("==> Load dataset ...")
        X, y = read_data(dataroot, debug, diff)
        train_size = 0.75

        runs += 1
        if debug:
            epochs = 2
        
        # Initialize model
        print("==> Initialize DA-RNN model ...")
        model = DA_rnn(
            X,
            y,
            ntimestep,
            nhidden_encoder,
            nhidden_decoder,
            batchsize,
            lr,
            epochs,
            loss_func,
            train_size,
        )


        run_name = f'batch_size={batchsize} lr={lr} epochs = {epochs} nhidden_encoder = {nhidden_encoder} nhidden_decoder = {nhidden_decoder}'
        run_name += f' ntimestep = {ntimestep} loss_func = {loss_func} diff = {diff}'

        if debug:
            run_name += ' Test'

        train_log_dir = 'logs/train/' + run_name
        train_summary_writer = summary.create_file_writer(train_log_dir)

        # Train
        print("==> Start training ...", runs)
        print(run_name)
        model.train(train_summary_writer)

        print("==> Testing ...")
        # Prediction
        y_pred = model.test()

        # Test statisitcs
        y_true = y[int(X.shape[0] * train_size):]
        err = np.abs(y_true - y_pred)
        mae = np.mean(err)
        rmse = np.sqrt(np.mean(err ** 2))
        print("Mean absolute error: ", mae, "Root mean Square error: ", rmse)

        run_stats.append((rmse, mae, run_name))

        # fig1 = plt.figure()
        # plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
        # plt.savefig("1.png")
        # plt.close(fig1)

        # fig2 = plt.figure()
        # plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
        # plt.savefig("2.png")
        # plt.close(fig2)

        # fig3 = plt.figure()
        # plt.plot(y_pred, label='Predicted')
        # plt.plot(model.y[model.train_timesteps:], label="True")
        # plt.legend(loc='upper left')
        # plt.savefig("3.png")
        # plt.close(fig3)
        # print('Finished Training')
    run_stats.sort()
    print("Best runs: ")
    for run in run_stats:
        print("Name:", run[2], "RMSE ", run[0], "MAE: ", run[1])

main()