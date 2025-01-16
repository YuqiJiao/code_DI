import math
import numpy as np
import torch
import tqdm
from sklearn.utils import check_random_state
from numpy.random import RandomState
from opendataval.dataval.api import DataEvaluator



from partition_ import Partition1, Partition2




class DIOOB(DataEvaluator):

    def __init__(self, outer_iter=800, port=0.1, mut=1.7, port_delet=0.3, random_state: RandomState = None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.data_val = None
        self.outer_iter = outer_iter
        self.port = port
        self.Mut = mut
        self.port_delet = port_delet
        self.random_state = check_random_state(random_state)

    def input_data(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_valid: torch.Tensor,
            y_valid: torch.Tensor,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid


        self.pre_num_points = len(x_train)

        return self



    def filter_noise(self,
                     batch_size=10, num_classe=2,
                     dataset = None,  clean_labels = None,
                     optimizer_fn=None, scheduler_fn=None,
                     ground_truth_clean=None,
                     data_perc=.12,
                     seed=None, device=None,
                     boost=100,  # 8，alpha
                     beta=4,
                     bag=3, *args, **kwargs):
        boost = max(boost, 1)
        bag = max(bag, 1)
        num_classes = num_classe
        active_ = np.ones(len(self.x_train))
        active = active_ == active_
        learners = []
        sets = []
        losses = np.zeros((len(self.x_train),))
        for b in range(1):
            active = active_ == active_
            for e in range(boost):
                if len(self.y_train[0]) <= 2:
                    if sum(active)/len(active) <= 0.025:
                        break
                else:
                    if sum(active)/len(active) <= 0.05:
                        break
                classifier_fn = self.pred_model.clone()

                dp = min(1, data_perc * len(active) / sum(active))
                if len(self.y_train[0]) <= 2:
                    wl = Partition1(
                        classifier_fn=classifier_fn,
                        x_train=self.x_train,
                        y_train=self.y_train,
                        active=active,
                        num_classes=num_classes,
                        batch_size=batch_size,
                        data_perc=dp,
                        refit_it=2 + max(1, min(3, int(1 // dp))),
                        expansion=1 / beta,
                        device=device
                    )
                    core_set, net = wl.get_core_set()
                    learners.append(net)
                    sets.append(core_set)
                else:
                    wl = Partition2(
                        classifier_fn=classifier_fn,
                        x_train=self.x_train,
                        y_train=self.y_train,
                        active=active,
                        num_classes=num_classes,
                        batch_size=batch_size,
                        data_perc=dp,
                        refit_it=2 + max(1, min(3, int(1 // dp))), b=b, losses=losses,
                        expansion=1 / beta,
                        device=device
                    )
                    core_set, net, loss = wl.get_core_set()
                    learners.append(net)
                    sets.append(core_set)
                    losses = loss

                del classifier_fn

                if boost > 1:
                    active = np.logical_and(active, ~core_set)

                if active.sum() == 0:

                    break

        sets.append(active)
        c_num = np.zeros(len(learners))

        v_bool = np.zeros(len(learners))

        for idx in range(len(learners)):
            for k in range(len(learners)):
                val_x = [self.x_train[i] for i in range(len(self.x_train)) if sets[k][i]]
                val_y = [self.y_train[i] for i in range(len(self.y_train)) if sets[k][i]]
                val_x = torch.stack(val_x)
                val_y = torch.stack(val_y)

                y_pred = learners[idx].predict(val_x)
                y_pred = torch.Tensor(y_pred)
                predicted = torch.argmax(y_pred, dim=1)
                predicted = predicted.cpu().numpy()

                val_y = [i.numpy() for i in val_y]
                val_y = np.stack(val_y)
                val_y = np.argmax(val_y, axis=1)

                if sum(predicted==val_y)>len(val_x)*0.5:
                    c_num[idx] += len(val_x)
        print(c_num)
        for idx_ in range(len(learners)):
            if c_num[idx_] > len(self.x_train)*0.5:
                v_bool[idx_] = 1

        return sets, v_bool




    def train_data_values(self, *args, **kwargs):
        num_class = len(self.y_train[0])

        sets, v_bool = self.filter_noise(num_classe=num_class)

        bool = v_bool
        bool = np.array(bool)
        if np.sum(bool) <= 0.5 * len(bool):
            for i in range(len(bool)):
                if i >= int(0.5 * len(bool)):
                    v_bool[i] = 1
        for i in range(len(v_bool)):
            sets[i] = sets[i] * (1-v_bool[i])

        keep = np.ones(len(self.x_train))
        for i in range(len(v_bool)):
            keep = keep - sets[i]
        num_keep = np.sum(keep)
        target_y = self.y_train
        target_y = torch.argmax(torch.Tensor(target_y), dim=1)
        target_y = target_y.cpu().numpy()
        net = None
        correct = None
        epochs = 5

        pre_bottom_20_indices = np.where(keep == 0)[0]
        pre_top_80_indices = np.where(keep != 0)[0]


        pre_split_point = int(len(pre_bottom_20_indices) * 1.0)


        weights = np.zeros((self.outer_iter, self.pre_num_points))
        distances = np.zeros((self.outer_iter, self.pre_num_points))
        scores = np.zeros((self.outer_iter, 1))
        for idx in tqdm.tqdm(range(self.outer_iter)):

            index_ = self.random_state.choice(int(self.pre_num_points - pre_split_point),
                                              size=int(self.pre_num_points - pre_split_point), replace=True)
            indices_in = pre_top_80_indices[index_]


            mask = np.isin(np.arange(int(self.pre_num_points - pre_split_point)), index_, invert=True)


            indices_out = np.arange(int(self.pre_num_points - pre_split_point))[mask]

            indices_out = pre_top_80_indices[indices_out]

            indices_out = np.concatenate((indices_out, pre_bottom_20_indices))
            curr_model = self.pred_model.clone()
            curr_model.fit(self.x_train[indices_in], self.y_train[indices_in], *args, **kwargs)
            y_train_hat = curr_model.predict(self.x_train[indices_out])
            y_valid = curr_model.predict(self.x_valid)
            score = self.evaluate(y_valid, self.y_valid)
            scores[idx] = score

            distance = torch.nn.functional.cross_entropy(y_train_hat, self.y_train[indices_out], reduction='none')
            distance = distance.detach().cpu().numpy()

            distances[idx, indices_out] = distance



            del curr_model


        sorted_indices = np.argsort(scores[:, 0])


        percentile = self.port_delet
        num_indices_to_select = int(len(sorted_indices) * percentile)


        selected_indices = sorted_indices[:num_indices_to_select]
        selected_indices = selected_indices.reshape(-1, 1)

        distances = np.delete(distances, selected_indices[:, 0], axis=0)


        non_zero_counts = np.count_nonzero(distances, axis=0)


        non_zero_sums = np.sum(distances, axis=0)


        fin_dis = non_zero_sums / non_zero_counts

        self.data_val = -fin_dis

    def evaluate_data_values(self) -> np.ndarray:
        return self.data_val
