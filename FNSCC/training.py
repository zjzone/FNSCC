import numpy as np
from sklearn.manifold import TSNE
from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader
import torch
import torch.nn as nn
from torch.nn import functional as F
from learner.cluster_utils import target_distribution
from learner.contrastive_utils import PairConLossWithNeighbors
from numpy import where
from sklearn import cluster
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta

        self.contrast_loss = PairConLossWithNeighbors(temperature=self.args.temperature,k=self.args.neighbor)  # 返回对比损失

        
        self.gstep = 0
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")
        
    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=self.args.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )

        return token_feat
        

    def prepare_transformer_input(self, batch):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)

            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)#batch*3*max_length
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)#batch*3*max_length
            
        elif len(batch) == 2:
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = self.get_batch_token(text)
            
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
            
        return input_ids.cuda(), attention_mask.cuda()

    def train_step_explicit(self, input_ids, attention_mask):
        #batch*3*max_length

        embd1, embd2, embd3 = self.model(input_ids, attention_mask, task_type="explicit")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        #
        losses = self.contrast_loss(feat1, feat2)
        #
        loss = self.eta * (losses["loss"])

        # Clustering loss
        if self.args.objective == "SCCL":
            output=self.model.update_cluster_prob_with_neighbors(embd1, self.args.neighbor, 0.5)
            target = target_distribution(output).detach() #(batch,num_classes)，

            cluster_loss = F.cross_entropy((output + 1e-08).log(), target)
            loss += cluster_loss
            losses["cluster_loss"] = cluster_loss.item()
        all_loss=loss
        contrast_loss = losses["loss"]
        #优化器反向传播优化模型
        loss.backward()
        # 加入梯度裁剪
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        #return contrast_loss
        return all_loss,contrast_loss,cluster_loss


    def train(self):
        print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))
        #scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.model.train()

        epoch_loss_list = []
        epoch_contrast_list = []
        epoch_cluster_list = []
        epoch_acc_list = []
        epoch_nmi_list=[]


        for i in np.arange(self.args.max_iter+1): #每次训练一个batch
            try:
                batch = next(train_loader_iter) #next() 函数用于从迭代器中获取下一个批次的数据
            except: #如果引发了异常，通常是因为数据加载器已经遍历完所有数据，进入了 StopIteration 异常，表示需要重新开始遍历数据集。
                train_loader_iter = iter(self.train_loader) #train_loader_iter 是 train_loader 的迭代器。
                batch = next(train_loader_iter)
            # if(len(batch['label'])>self.args.neighbor):
            #     continue
            input_ids, attention_mask = self.prepare_transformer_input(batch) #大小均为batch*3*max_length，其中3表示text，text1，text2
            #计算损失，优化模型
            all_loss,contrast_loss,cluster_loss = self.train_step_virtual(input_ids, attention_mask) if self.args.augtype == "virtual" else self.train_step_explicit(input_ids, attention_mask)
            #contrast_loss = self.train_step_virtual(input_ids,attention_mask) if self.args.augtype == "virtual" else self.train_step_explicit(input_ids, attention_mask)
            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==self.args.max_iter)):
                print("[{}]-----".format(i))
                epoch_loss = all_loss.detach().to('cpu').numpy()
                epoch_contrast_loss = contrast_loss.detach().to('cpu').numpy()
                epoch_cluster_loss = cluster_loss.detach().to('cpu').numpy()
                epoch_loss_list.append(epoch_loss)
                epoch_contrast_list.append(epoch_contrast_loss)
                epoch_cluster_list.append(epoch_cluster_loss)
                print("all_loss", epoch_loss)
                print("contrast_loss", epoch_contrast_loss)
                print("cluster_loss", epoch_cluster_loss)

                #statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                acc_evl,nmi_evl=self.evaluate_embedding(i)
                epoch_acc_list.append(acc_evl)
                epoch_nmi_list.append(nmi_evl)

                self.model.train()
                #scheduler.step()
        # epochs = range(1, len(epoch_loss_list))
        # 创建一个 1 行 2 列的子图布局
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # 子图1：聚类损失和对比损失和总损失
        # 绘制折线图
        #ax1.plot(epochs, epoch_loss_list[:-1], marker='o', color='b', label='All_Loss', markersize=2)
        #ax1.plot(epochs, epoch_contrast_list[:-1], marker='x', color='g', label='Contrast_Loss', markersize=3)
        #ax1.plot(epochs, epoch_cluster_list[:-1], marker='s', color='r', label='Cluster_Loss', markersize=3)

        # xticks = list(range(1, len(epoch_loss_list),2))  # 原始坐标
        # xtick_labels = [100 * x for x in xticks]  # 修改后的坐标值
        # ax1.set_xticks(xticks)
        # ax1.set_xticklabels(xtick_labels)

        # 添加标题和标签
        # ax1.set_title('Loss Curve Over Epochs')
        # ax1.set_xlabel('Iterations', fontsize=14)
        # ax1.set_ylabel('Loss', fontsize=14)
        # 显示图例
        # ax1.legend()

        # 子图2：聚类acc
        # ax2.plot(epochs, epoch_acc_list[:-1], marker='o', color='b', label='ACC', markersize=3)
        # ax2.plot(epochs, epoch_nmi_list[:-1], marker='x', color='g', label='NMI', markersize=3)
        # ax2.set_xticks(xticks)
        # ax2.set_xticklabels(xtick_labels)
        # # ax2.set_title('Clustering')
        # ax2.set_xlabel('Iterations', fontsize=14)
        # ax2.set_ylabel('Performance', fontsize=14)
        # ax2.legend()
        # 保存图像为文件
        # plt.savefig('loss_acc_curve.pdf')  # 保存为 PNG 文件
        return None
    def evaluate_embedding(self, step):
        dataloader = unshuffle_loader(self.args)
        print('---- {} evaluation batches ----'.format(len(dataloader)))
        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label'] 
                feat = self.get_batch_token(text)
                embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")

                model_prob = self.model.get_cluster_prob(embeddings)
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_prob = torch.cat((all_prob, model_prob), dim=0)
                    
        # Initialize confusion matrices
        confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)
        
        all_pred = all_prob.max(1)[1]  #模型预测标签
        confusion_model.add(all_pred, all_labels)
        confusion_model.optimal_assignment(self.args.num_classes)
        acc_model = confusion_model.acc()
        kmeans = cluster.KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)

        # 对embedding归一化
        # all_embeddings = F.normalize(all_embeddings, p=2, dim=-1)
        embeddings = all_embeddings.cpu().numpy()

        kmeans.fit(embeddings)
        pred_labels = torch.tensor(kmeans.labels_.astype(np.int_)) #聚类预测标签

        #可视化
        # label = pred_labels.numpy()
        # if True:
        #     tsne = TSNE(n_components=2, metric="euclidean", random_state=0)  # Tsne可视化降维到2维
        #     X=tsne.fit_transform(embeddings)
        #     for cluster_label in np.unique(label):
        #         # 获取此群集的示例的行索引
        #         row_ix = where(label == cluster_label)
        #         # 创建这些样本的散布
        #         if cluster_label == 1:
        #             c = "b"
        #         elif cluster_label== 2:
        #             c = "g"
        #         elif cluster_label == 3:
        #             c = "y"
                # elif cluster_label == 4:
                #     c = "c"
                # elif cluster_label == 5:
                #     c = "m"  # 紫
                # elif cluster_label == 6:
                #     c = "r"  # 紫
                # elif cluster_label == 7:
                #     c = "k"  # 紫
                # else:
                #     c = "#FF5733"
                # plt.scatter(X[row_ix, 0], X[row_ix, 1], s=2, c=c)

            #plt.title("t-SNE Clustering Visualization")
            # plt.savefig("./Image/"+str(step)+"tsne_visualization.pdf")
            # plt.close()

        
        # clustering accuracy 
        confusion.add(pred_labels, all_labels)
        confusion.optimal_assignment(self.args.num_classes)
        acc = confusion.acc()

        ressave = {"acc":acc, "acc_model":acc_model}
        ressave.update(confusion.clusterscores())
        for key, val in ressave.items():
            self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)
            
        np.save(self.args.resPath + 'acc_{}.npy'.format(step), ressave)
        np.save(self.args.resPath + 'scores_{}.npy'.format(step), confusion.clusterscores())
        np.save(self.args.resPath + 'mscores_{}.npy'.format(step), confusion_model.clusterscores())
        # np.save(self.args.resPath + 'mpredlabels_{}.npy'.format(step), all_pred.cpu().numpy())
        # np.save(self.args.resPath + 'predlabels_{}.npy'.format(step), pred_labels.cpu().numpy())
        # np.save(self.args.resPath + 'embeddings_{}.npy'.format(step), embeddings)
        # np.save(self.args.resPath + 'labels_{}.npy'.format(step), all_labels.cpu())

        print('[Representation] Clustering scores:',confusion.clusterscores()) 
        print('[Representation] ACC: {:.5f}'.format(acc))
        print('[Model] Clustering scores:',confusion_model.clusterscores()) 
        print('[Model] ACC: {:.5f}'.format(acc_model))
        return acc,confusion.clusterscores()['NMI']



             