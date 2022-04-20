from abc import ABC, abstractmethod
import torch
import time


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit_iterator_one_epoch(self, iterator):
        pass

    @abstractmethod
    def fit_batch(self, iterator):
        pass

    @abstractmethod
    def evaluate_iterator(self, iterator):
        pass

    def update_from_model(self, model):
        """
        update parameters using gradients from another model
        :param model: Model() object, gradients should be precomputed;
        """
        for param_idx, param in enumerate(self.net.parameters()):
            param.grad = list(model.net.parameters())[param_idx].grad.data.clone()

        self.optimizer.step()
        self.lr_scheduler.step()

    def fit_batches(self, iterator, n_steps):
        global_loss = 0
        global_acc = 0

        for step in range(n_steps):
            batch_loss, batch_acc = self.fit_batch(iterator)
            global_loss += batch_loss
            global_acc += batch_acc

        return global_loss / n_steps, global_acc / n_steps

    def fit_iterator(self, train_iterator, val_iterator=None, n_epochs=1, path=None, verbose=0):
        best_valid_loss = float('inf')

        for epoch in range(n_epochs):

            start_time = time.time()

            train_loss, train_acc = self.fit_iterator_one_epoch(train_iterator)
            if val_iterator:
                valid_loss, valid_acc = self.evaluate_iterator(val_iterator)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if val_iterator:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if path:
                        torch.save(self.net, path)

            if verbose:
                print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
                if val_iterator:
                    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    def get_param_tensor(self):
        param_list = []

        for param in self.net.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)
