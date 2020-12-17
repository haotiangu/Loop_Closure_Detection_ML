
import numpy as np
from sklearn.neighbors import KDTree
import loss.pointnetvlad_loss as PNV_loss
import models.PointNetVlad as PNV
import torch
import torch.nn as nn
from loading_pointclouds import *
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from config import args, cfg



class TrainDataset(Dataset):

    def __init__(self, training_queries):
        super(TrainDataset, self).__init__()
        self.training_queries = training_queries
        self.n_item = len(training_queries)
        self.__get_transform_dict()
        
    def __get_transform_dict(self):
        self.transform_dict = {}
        tv = 0
        for k, v in self.training_queries.items():
            if len(v['positives']) > cfg.TRAIN_POSITIVES_PER_QUERY:
                self.transform_dict[tv] = k
                tv += 1
                

    def __getitem__(self, item):
        
        key = self.transform_dict[item]
        queries, positives, negatives,other_neg = \
            get_query_tuple(self.training_queries[key],
                            cfg.TRAIN_POSITIVES_PER_QUERY, 
                            cfg.TRAIN_NEGATIVES_PER_QUERY,
                            self.training_queries, hard_neg=[], other_neg=True)
        queries = torch.tensor(queries, dtype=torch.float32)
        positives = torch.tensor(positives, dtype=torch.float32)
        negatives = torch.tensor(negatives, dtype=torch.float32)
        other_neg = torch.tensor(other_neg, dtype=torch.float32)
        queries = queries.unsqueeze(0)
        other_neg = other_neg.unsqueeze(0)
        return queries, positives, negatives, other_neg

    def __len__(self):
        return len(self.transform_dict)

class EvalDastaSet(Dataset):
    def __init__(self):
        self.database_sets = get_sets_dict(cfg.EVAL_DATABASE_FILE)[:2]
        self.query_sets = get_sets_dict(cfg.EVAL_QUERY_FILE)[:2]


    def __getitem__(self, item):
        return self.database_sets[item], self.query_sets[item]

    def __len__(self):
        return len(self.database_sets)

class Trainer(pl.LightningModule):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.model = PNV.Model(global_feat=True, feature_transform=True,
                                 max_pool=False, output_dim=cfg.FEATURE_OUTPUT_DIM, num_points=cfg.NUM_POINTS)
        self.loss_function = self.get_loss_func()

    def training_step(self, batch, batch_ix):
        queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor = batch
        print(queries_tensor.device)
        feed_tensor = torch.cat(
        (queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
        feed_tensor = feed_tensor.view((-1, 1, cfg.NUM_POINTS, 3))
        output = self.model(feed_tensor)
        output = output.view(cfg.BATCH_NUM_QUERIES, -1, cfg.FEATURE_OUTPUT_DIM)
    
        output_queries, output_positives, output_negatives, output_other_neg = torch.split(
        output, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=1)
        loss = self.loss_function(output_queries, output_positives,
                             output_negatives, output_other_neg, 
                             cfg.MARGIN_1, cfg.MARGIN_2, 
                             use_min=cfg.TRIPLET_USE_BEST_POSITIVES, 
                             lazy=cfg.LOSS_LAZY, ignore_zero_loss=cfg.LOSS_IGNORE_ZERO_BATCH)
        self.log('train loss', loss)
        return loss

    def get_loss_func(self,):
        if cfg.LOSS_FUNCTION == 'quadruplet':
            loss_function = PNV_loss.quadruplet_loss
        else:
            loss_function = PNV_loss.triplet_loss_wrapper
        return loss_function

    def configure_optimizers(self,):
        optimizer = torch.optim.Adam(self.model.parameters(), self.args.learning_rate)
        return optimizer

    def validation_step(self, batch, batch_ix):
        dict_database, dict_query = batch
        latent_database = self.get_latent_vectors(dict_database)
        latent_query = self.get_latent_vectors(dict_query)
        return latent_database, latent_query

    def validation_epoch_end(self, outputs) :
        database_vectors = [i[0] for i in outputs]
        query_vectors = [i[1] for i in outputs]
        recall = np.zeros(25)
        count = 0
        similarity = []
        one_percent_recall = []
        query_sets = self.val_dataloader().query_sets

        for m in range(len(query_sets)):
            for n in range(len(query_sets)):
                if (m == n):
                    continue
                pair_recall, pair_similarity, pair_opr = self.get_recall(
                    m, n, database_vectors, query_vectors, query_sets)
                recall += np.array(pair_recall)
                count += 1
                one_percent_recall.append(pair_opr)
                for x in pair_similarity:
                    similarity.append(x)

        ave_recall = recall / count
        average_similarity = np.mean(similarity)
        ave_one_percent_recall = np.mean(one_percent_recall)
        self.log('test_ave_recall', ave_recall, prog_bar=True,
                 on_epoch=True)
        self.log('test_average_similarity', average_similarity, prog_bar=True,
                 on_epoch=True)
        self.log('test_ave_one_percent_recall', ave_one_percent_recall, prog_bar=True,
                 on_epoch=True)

    def get_recall(self, m, n, DATABASE_VECTORS, QUERY_VECTORS, oxford_evaluation_query.pickle):

        database_output = DATABASE_VECTORS[m]
        queries_output = QUERY_VECTORS[n]


        database_nbrs = KDTree(database_output)

        num_neighbors = 25
        recall = [0] * num_neighbors

        top1_similarity_score = []
        one_percent_retrieved = 0
        threshold = max(int(round(len(database_output) / 100.0)), 1)

        num_evaluated = 0
        for i in range(len(queries_output)):
            true_neighbors = QUERY_SETS[n][i][m]
            if (len(true_neighbors) == 0):
                continue
            num_evaluated += 1
            distances, indices = database_nbrs.query(
                np.array([queries_output[i]]), k=num_neighbors)
            for j in range(len(indices[0])):
                if indices[0][j] in true_neighbors:
                    if (j == 0):
                        similarity = np.dot(
                            queries_output[i], database_output[indices[0][j]])
                        top1_similarity_score.append(similarity)
                    recall[j] += 1
                    break

            if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
                one_percent_retrieved += 1

        one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
        recall = (np.cumsum(recall) / float(num_evaluated)) * 100
        return recall, top1_similarity_score, one_percent_recall

    def get_latent_vectors(self, dict_to_process):
        train_file_idxs = np.arange(0, len(dict_to_process.keys()))

        batch_num = cfg.EVAL_BATCH_SIZE * \
                    (1 + cfg.EVAL_POSITIVES_PER_QUERY + cfg.EVAL_NEGATIVES_PER_QUERY)
        q_output = []
        for q_index in range(len(train_file_idxs) // batch_num):
            file_indices = train_file_idxs[q_index *
                                           batch_num:(q_index + 1) * (batch_num)]
            file_names = []
            for index in file_indices:
                file_names.append(dict_to_process[index]["query"])
            queries = load_pc_files(file_names)

            with torch.no_grad():
                feed_tensor = torch.from_numpy(queries).float()
                feed_tensor = feed_tensor.unsqueeze(1)
                feed_tensor = feed_tensor.to(cfg.device)
                out = self.model(feed_tensor)

            out = out.squeeze().cpu().numpy()
            q_output.append(out)
        q_output = np.concatenate(q_output, 0)
        return q_output




if __name__ == "__main__":
    # Load dictionary of training queries
    TRAINING_QUERIES = get_queries_dict(cfg.TRAIN_FILE)
    evaldataset = EvalDastaSet()
    dataset = TrainDataset(TRAINING_QUERIES)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = Trainer(args)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloader=dataloader, val_dataloaders=evaldataset)
    # trainer.test(model, evaldataset)
