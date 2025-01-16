import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Subset
from torch.cuda.amp import autocast, GradScaler

from opendataval.dataval import DataOob

from opendataval.metrics import Metrics

from opendataval.dataval.api import DataEvaluator





class Partition1(object):#object, DataEvaluator
    """
    Weak Learner with ISPL objective
    """

    def __init__(self, classifier_fn, x_train, y_train, active,
                 num_classes, batch_size,
                 data_perc,
                 refit_it,
                 optimizer_fn = None, criterion = None, expansion=.25,
                 scheduler_fn=None,
                 ground_truth_clean=None, clean_labels=None,
                 device=None, random=None, amp=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier_fn = classifier_fn

        # self.dataset = dataset
        self.x_train = x_train
        self.y_train = y_train
        self.active = active
        # self.num_classes = num_classes
        self.num_classes = len(y_train[0])
        # self.dataset_size = len(dataset)
        self.dataset_size = len(x_train)


        self.batch_size = batch_size
        self.data_perc = data_perc
        self.refit_it = refit_it

        self.optimizer_fn = optimizer_fn
        self.criterion = criterion
        self.scheduler_fn = scheduler_fn

        self.ground_truth_clean = ground_truth_clean
        self.clean_labels = clean_labels
        self.amp = amp #True

        self.warm_up = 8
        self.retrain_epochs = 40

        self.loss_m = None
        self.decay = .5
        self.expansion = expansion

        self.device = device if device is not None else torch.device("cpu")

        self.random = random

    def get_core_set(self, *args, **kwargs):

        indices_to_keep = np.copy(self.active)

########################################################################################################
        indices_to_keep = self.active != self.active
        indices_to_keep = np.array(indices_to_keep)
########################################################################################################


        val_x = [self.x_train[i] for i in range(len(self.x_train)) if self.active[i]]
        val_y = [self.y_train[i] for i in range(len(self.y_train)) if self.active[i]]
        val_x = torch.stack(val_x)
        val_y = torch.stack(val_y)



        if self.data_perc >= .999:
            self.refit_it = 0
            active_keep = self.active[self.active]

        if len(self.y_train[0]) > 2:
            nums = 700*0.07
        else:
            nums = 1000*0.07

        beta = 1.0
        active_ = np.copy(self.active)
        num = np.arange(0,len(self.x_train))
        active_num = sum(self.active)

        for it in range(1):
            print(f"refit it {it}")
            beta = beta - 0.05



            train_x = [self.x_train[i] for i in range(len(self.x_train)) if active_[i] and self.active[i]]
            train_y = [self.y_train[i] for i in range(len(self.y_train)) if active_[i] and self.active[i]]
            train_x = torch.stack(train_x)
            train_y = torch.stack(train_y)
            oob = DataOob(num_models=100, random_state=self.random).input_data(x_train=train_x,
                                                                                  y_train=train_y,
                                                                                  x_valid=val_x,
                                                                                  y_valid=val_y, ).input_model(
                pred_model=self.classifier_fn.clone()).input_metric(
                metric=Metrics("accuracy"))
            oob.train_data_values()
            oob_val = oob.evaluate_data_values()
            oob_val = np.array(oob_val)

            losses = np.zeros((len(self.x_train),))
            a=1
            result = self.active & active_
            losses[result] = oob_val
            losses[~self.active] = 1000

            predicted=[]




            active_keep = \
                self.get_trimmed_set(
                    it,
                    losses,
                    predicted,
                    indices_to_keep[self.active],
                    val_y, beta, active_num)#active_labels
            indices_to_keep = np.copy(self.active)
            active_train = np.copy(active_keep)
            active_num = sum(active_train)



            active_ = np.copy(active_keep)
            ##############################################################################################################
            train_keep = indices_to_keep[self.active]
            print("==================================")
            print(active_keep.sum())
            print(len(active_keep) * 0.07)
            if len(self.y_train[0]) <= 2:
                if active_keep.sum() <= 0.025*len(self.x_train):
                    print("break")
                    print(active_keep.sum())
                    print(len(active_keep) * 0.07)
                    break
            else:
                if active_keep.sum() <= 0.2*len(self.x_train):
                    print("break")
                    print(active_keep.sum())
                    print(len(active_keep) * 0.07)
                    break



        results = self.active & active_keep
        indices_to_keep = np.copy(results)

        if indices_to_keep.sum()==0:
            print('***********************************************************************************************')
            print('0')
        train_x = [self.x_train[i] for i in range(len(self.x_train)) if self.active[i] and active_keep[i]]
        train_y = [self.y_train[i] for i in range(len(self.y_train)) if self.active[i] and active_keep[i]]
        train_y = torch.stack(train_y)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('sets',indices_to_keep.sum())

        net = self.classifier_fn.clone()
        net.fit(train_x, train_y, *args, **kwargs)



        return indices_to_keep, net

    def retrain(self, indices_to_keep, net=None, epochs=None):
        if epochs is None:
            epochs = self.refit_it * 2
        if net is None:
            net = self.classifier_fn().to(self.device)
        if self.amp:
            scaler = GradScaler()
        optimizer = \
            self.optimizer_fn(net.parameters())
        if self.scheduler_fn is not None:
            scheduler = self.scheduler_fn(optimizer, epochs)
        else:
            scheduler = None
        filtered = Subset(self.dataset,
            [i for i in range(self.dataset_size) if self.active[i] and indices_to_keep[i]])

        trainloader = torch.utils.data.DataLoader(
            filtered, batch_size=self.batch_size,
            shuffle=True, num_workers=2, drop_last=True)

        self.train(trainloader, net, optimizer, scaler, scheduler, epochs)
        return net

    def get_trimmed_set(self, it,
            losses, predicted, indices_to_keep, labels, beta, active_num):

        m_losses = np.mean(losses)
        keep1 = losses <= m_losses
        if keep1.sum()==0:
            print('*******************************************keep1**********************************************************')

        keep2 = losses != losses

        sorted_losses = np.argsort(losses)

        numm_to_keep = int(self.active.sum() * beta)

        if len(self.y_train[0]) <= 2:
            keep2[sorted_losses[:int(0.025*len(self.x_train))]] = True
        else:
            keep2[sorted_losses[:int(0.2*len(self.x_train))]] = True

        if sum(keep1) <= sum(keep2):
            keep = keep2
        else:
            keep = keep2

        return keep

    def train(self, trainloader, net, optimizer, scaler, scheduler, num_epochs=10):
        net.train()
        num_batches = len(trainloader)
        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = \
                    inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                if self.amp:#True
                    with autocast():
                        outputs = net(inputs)
                        loss = self.criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = net(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            if scheduler is not None:
                scheduler.step()

        return net

    def test(self, valloader, net, loss_fn=None):
        if loss_fn is None: loss_fn = nn.CrossEntropyLoss(reduction="none")

        losses = []
        predicted = []

        net.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valloader):
                inputs, labels = \
                    inputs.to(self.device), labels.to(self.device)

                outputs = net(inputs)
                _, pred = torch.max(outputs.data, 1)
                predicted.append(pred)

                loss = loss_fn(outputs, labels)
                losses.append(loss)

        losses = torch.cat(losses).cpu().numpy()
        predicted = torch.cat(predicted).cpu().numpy()
        return losses, predicted


class Partition2(object):#object, DataEvaluator
    """
    Weak Learner with ISPL objective
    """

    def __init__(self, classifier_fn, x_train, y_train, active,
                 num_classes, batch_size,
                 data_perc,
                 refit_it, b, losses,
                 optimizer_fn = None, criterion = None, expansion=.25,
                 scheduler_fn=None,
                 ground_truth_clean=None, clean_labels=None,
                 device=None, random=None, amp=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier_fn = classifier_fn

        self.x_train = x_train
        self.y_train = y_train
        self.active = active
        print('self.active=', sum(self.active))
        self.num_classes = len(y_train[0])
        self.dataset_size = len(x_train)
        self.losses = losses


        self.batch_size = batch_size
        self.data_perc = data_perc
        self.refit_it = refit_it

        self.optimizer_fn = optimizer_fn
        self.criterion = criterion
        self.scheduler_fn = scheduler_fn

        self.ground_truth_clean = ground_truth_clean
        self.clean_labels = clean_labels
        self.amp = amp

        self.warm_up = 8
        self.retrain_epochs = 40

        self.loss_m = None
        self.decay = .5
        self.expansion = expansion

        self.device = device if device is not None else torch.device("cpu")

        self.random = random

    def get_core_set(self, *args, **kwargs):

        indices_to_keep = self.active != self.active
        indices_to_keep = np.array(indices_to_keep)

        val_x = [self.x_train[i] for i in range(len(self.x_train)) if self.active[i]]
        val_y = [self.y_train[i] for i in range(len(self.y_train)) if self.active[i]]
        val_x = torch.stack(val_x)
        val_y = torch.stack(val_y)

        if self.data_perc >= .999:
            self.refit_it = 0
            active_keep = self.active[self.active]

        if len(self.y_train[0]) > 2:
            nums = 700*0.07
        else:
            nums = 1000*0.07

        beta = 1.0
        active_ = np.copy(self.active)
        num = np.arange(0,len(self.x_train))
        active_num = sum(self.active)

        for it in range(1):
            beta = beta - 0.05

            train_x = [self.x_train[i] for i in range(len(self.x_train)) if active_[i] and self.active[i]]
            train_y = [self.y_train[i] for i in range(len(self.y_train)) if active_[i] and self.active[i]]
            train_x = torch.stack(train_x)
            train_y = torch.stack(train_y)
            oob = DataOob(num_models=100, random_state=self.random).input_data(x_train=train_x,
                                                                                  y_train=train_y,
                                                                                  x_valid=val_x,
                                                                                  y_valid=val_y, ).input_model(
                pred_model=self.classifier_fn.clone()).input_metric(
                metric=Metrics("accuracy"))
            oob.train_data_values()
            oob_val = oob.evaluate_data_values()
            oob_val = np.array(oob_val)

            result = self.active & active_

            self.losses[result] = oob_val
            self.losses[~self.active] += 1

            predicted=[]
            labels = np.argmax(self.y_train, axis=1)
            label_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
            np.save('label_indices.npy', label_indices)

            active_keep = \
                self.get_trimmed_set(
                    it,
                    self.losses,
                    predicted,
                    indices_to_keep[self.active],
                    labels, beta, active_num, label_indices)
            indices_to_keep = np.copy(self.active)
            active_train = np.copy(active_keep)
            active_num = sum(active_train)
            self.losses[result] = 1 - self.losses[result]

            active_ = np.copy(active_keep)

            train_keep = indices_to_keep[self.active]
            if len(self.y_train[0]) <= 2:
                if active_keep.sum() <= 0.025*len(self.x_train):
                    print("break")
                    print(active_keep.sum())
                    print(len(active_keep) * 0.07)
                    break
            else:
                if active_keep.sum() <= 0.075*len(self.x_train):
                    print("break")
                    print(active_keep.sum())
                    print(len(active_keep) * 0.07)
                    break



        results = self.active & active_keep
        indices_to_keep = np.copy(results)
        train_x = [self.x_train[i] for i in range(len(self.x_train)) if active_keep[i]]
        train_y = [self.y_train[i] for i in range(len(self.y_train)) if active_keep[i]]
        train_y = torch.stack(train_y)
        net = self.classifier_fn.clone()
        net.fit(train_x, train_y, *args, **kwargs)
        loss = self.losses


        return indices_to_keep, net, loss

    def retrain(self, indices_to_keep, net=None, epochs=None):
        if epochs is None:
            epochs = self.refit_it * 2
        if net is None:
            net = self.classifier_fn().to(self.device)
        if self.amp:
            scaler = GradScaler()
        optimizer = \
            self.optimizer_fn(net.parameters())
        if self.scheduler_fn is not None:
            scheduler = self.scheduler_fn(optimizer, epochs)
        else:
            scheduler = None
        filtered = Subset(self.dataset,
            [i for i in range(self.dataset_size) if self.active[i] and indices_to_keep[i]])

        trainloader = torch.utils.data.DataLoader(
            filtered, batch_size=self.batch_size,
            shuffle=True, num_workers=2, drop_last=True)

        self.train(trainloader, net, optimizer, scaler, scheduler, epochs)
        return net

    def get_trimmed_set(self, it,
            losses, predicted, indices_to_keep, labels, beta, active_num, label_indices):

        m_losses = np.mean(losses)
        keep1 = losses <= m_losses
        keep2 = losses != losses
        if len(self.y_train[0]) > 2:

            for label in np.unique(labels):
                label = label
                indices = label_indices[label]
                losses_for_label = losses[indices]
                sorted_indices_for_label = np.argsort(losses_for_label)
                min_025_indices = indices[
                    sorted_indices_for_label[:int(0.01 * len(sorted_indices_for_label))]]

                keep2[min_025_indices] = True
        else:

            sorted_losses = np.argsort(losses)

            numm_to_keep = int(self.active.sum() * beta)
            r = 0.025
            if len(self.y_train[0]) <= 2:
                keep2[sorted_losses[:int(r*len(self.x_train))]] = True
            else:
                keep2[sorted_losses[:int(r*len(self.x_train))]] = True#0.2

        if sum(keep1) <= sum(keep2):
            keep = keep2
        else:
            keep = keep2
        true_indices = np.where(keep)[0]

        return keep

    def train(self, trainloader, net, optimizer, scaler, scheduler, num_epochs=10):
        net.train()
        num_batches = len(trainloader)
        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = \
                    inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                if self.amp:#True
                    with autocast():
                        outputs = net(inputs)
                        loss = self.criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = net(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            if scheduler is not None:
                scheduler.step()

        return net

    def test(self, valloader, net, loss_fn=None):
        if loss_fn is None: loss_fn = nn.CrossEntropyLoss(reduction="none")

        losses = []
        predicted = []

        net.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valloader):
                inputs, labels = \
                    inputs.to(self.device), labels.to(self.device)

                outputs = net(inputs)
                _, pred = torch.max(outputs.data, 1)
                predicted.append(pred)

                loss = loss_fn(outputs, labels)
                losses.append(loss)

        losses = torch.cat(losses).cpu().numpy()
        predicted = torch.cat(predicted).cpu().numpy()
        return losses, predicted
